#!/usr/bin/env python
# coding: utf-8

import argparse
import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from torch_geometric.datasets import Coauthor, Amazon

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

import tqdm
import sklearn.metrics

OGB_DIR = "../data/OGB"
DGL_DIR = "../data/DGL"
PYG_DIR = "../data/PYG"

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(mfgs[1], (h, h_dst))  # <---
        return h
    
def get_accuracy(model, dataloader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for _, _, mfgs in dataloader:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    return accuracy
''
def convert_pyg(dataset):
    # TODO
    data = dataset[0]
    num_nodes = data.num_nodes
    graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
    graph.ndata['feat'] = data.x
    graph.ndata['label'] = data.y
    if 'train_mask' in data.keys:
        train_nids = torch.tensor(range(graph.num_nodes()))[graph.ndata['train_mask']]
        valid_nids = torch.tensor(range(graph.num_nodes()))[graph.ndata['val_mask']]
        test_nids = torch.tensor(range(graph.num_nodes()))[graph.ndata['test_mask']]
    else:
        splits = [4,3,3]
        ratios = [np.sum(splits[:i]) / np.sum(splits) for i in range(len(splits) + 1)]
        # print(ratios)
        i = 0
        train_nids = torch.tensor(range(int(ratios[i] * num_nodes), int(ratios[i+1] * num_nodes)))
        i += 1
        valid_nids = torch.tensor(range(int(ratios[i] * num_nodes), int(ratios[i+1] * num_nodes)))
        i += 1
        test_nids  = torch.tensor(range(int(ratios[i] * num_nodes), int(ratios[i+1] * num_nodes)))
    return graph, train_nids, valid_nids, test_nids

def load_data(graph_name):
    if graph_name.lower().startswith('ogb'):
        dataset = DglNodePropPredDataset(graph_name.lower(), root=OGB_DIR)
        graph, node_labels = dataset[0]
        graph.ndata['label'] = node_labels[:, 0]
        idx_split = dataset.get_idx_split()
        train_nids = idx_split['train']
        valid_nids = idx_split['valid']
        test_nids = idx_split['test']
    elif graph_name in ['CS', 'Physics']:
        dataset = Coauthor(root=PYG_DIR, name=graph_name)
        return convert_pyg(dataset)
    elif graph_name in ['Computers', 'Photo']:
        dataset = Amazon(root=PYG_DIR, name=graph_name)
        return convert_pyg(dataset)
    else:
        if graph_name.lower().startswith('reddit'):
            dataset = dgl.data.RedditDataset(raw_dir=DGL_DIR)
        elif graph_name.lower().startswith('cora'):
            dataset = dgl.data.CoraGraphDataset(raw_dir=DGL_DIR)
        elif graph_name.lower().startswith('pubmed'):
            dataset = dgl.data.PubmedGraphDataset(raw_dir=DGL_DIR)
        elif graph_name.lower().startswith('citeseer'):
            dataset = dgl.data.CiteseerGraphDataset(raw_dir=DGL_DIR)
        else:
            raise NotImplementedError()
        graph = dataset[0]
        train_nids = torch.tensor(range(graph.num_nodes()))[graph.ndata['train_mask']]
        valid_nids = torch.tensor(range(graph.num_nodes()))[graph.ndata['val_mask']]
        test_nids = torch.tensor(range(graph.num_nodes()))[graph.ndata['test_mask']]
    graph = dgl.add_reverse_edges(graph)
    graph = dgl.add_self_loop(graph)
    print(graph)
    return graph, train_nids, valid_nids, test_nids

def main(args):
    SEED = 24
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    device = 'cuda:{}'.format(args.device)
    # graph, train_nids, valid_nids, test_nids = load_data('ogbn-arxiv')
    graph, train_nids, valid_nids, test_nids = load_data(args.dataset)

    num_features = graph.ndata['feat'].shape[1]
    num_classes = (graph.ndata['label'].max() + 1).item()
    print('Number of classes:', num_classes)

    # TODO: adapt model params accordingly
    num_layers = 2 
    fanouts = [2,4] # [5, 10]
    hidden_dim = 128
    batch_size = args.batch_size # 1024 # 256

    # if dgl.__version__.startswith('0.9'):
    #     sampler = dgl.dataloading.NeighborSampler(fanouts)
    # else:
    #     sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts[::-1])
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=batch_size,    # Batch size
        shuffle=True,       # TODO: Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    train_dataloader_no_shuffle = dgl.dataloading.DataLoader(
        # The following arguments are specific to DGL's DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=batch_size,    # Batch size
        shuffle=False,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    model = Model(num_features, hidden_dim, num_classes).to(device)

    opt = torch.optim.Adam(model.parameters())

    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, 
        dgl.dataloading.MultiLayerFullNeighborSampler(num_layers),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=device
    )

    test_dataloader = dgl.dataloading.DataLoader(
        graph, test_nids,
        dgl.dataloading.MultiLayerFullNeighborSampler(num_layers), 
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=device # torch.device('cpu')
    )

    # training_loop
    records = []
    stop_count = 0
    early_stop = args.early_stop # 10
    num_epochs = args.epochs     # 100

    # TODO: change shuffle mode
    # mode = 'shuffle'
    # mode = 'no shuffle'
    # mode = 'interleave'
    mode = args.shuffle
    switch_epoch = args.switch

    best_accuracy = 0
    best_model_path = 'model.pt'
    for epoch in tqdm.tqdm(range(num_epochs)):

        if mode == 'shuffle':
            curr_dataloader = train_dataloader
        elif mode == 'no shuffle':
            curr_dataloader = train_dataloader_no_shuffle
        elif mode == 'interleave':
            if epoch % 2 == 0:
                curr_dataloader = train_dataloader
            else:
                curr_dataloader = train_dataloader_no_shuffle
        elif mode == 'on & off':
            if epoch < switch_epoch:
                curr_dataloader = train_dataloader
            else:
                curr_dataloader = train_dataloader_no_shuffle
        elif mode == 'off & on':
            if epoch < switch_epoch:
                curr_dataloader = train_dataloader_no_shuffle
            else:
                curr_dataloader = train_dataloader
        else:
            raise NotImplementedError
        
        model.train()
        
        for _, _, mfgs in tqdm.tqdm(curr_dataloader, leave=False):
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']

            predictions = model(mfgs, inputs)

            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

    # with tqdm.tqdm(train_dataloader) as tq:
    #     for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
    #         # feature copy from CPU to GPU takes place here
    #         inputs = mfgs[0].srcdata['feat']
    #         labels = mfgs[-1].dstdata['label']

    #         predictions = model(mfgs, inputs)

    #         loss = F.cross_entropy(predictions, labels)
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #         accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

    #         tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)


        model.eval()

        predictions = []
        labels = []
    #     with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
    #         for input_nodes, output_nodes, mfgs in tq:
        with torch.no_grad():
            for _, _, mfgs in valid_dataloader:
                inputs = mfgs[0].srcdata['feat']
                labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    #         print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
            if best_accuracy < accuracy:
                stop_count = 0
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
                # test_accuracy = get_accuracy(model, test_dataloader)
                # records.append((epoch, accuracy, test_accuracy))
                records.append((epoch, accuracy))
                # print(records[-1])
            else:
                stop_count += 1
                if stop_count == early_stop:
                    break
               
    best_epoch = -1
    for item in records:
        # print(item)
        if item[1] == best_accuracy:
            best_epoch = item[0]
    print('Epoch {} Best Validation Accuracy {:.5f}'.format(best_epoch, best_accuracy))
    model = Model(num_features, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    acc = get_accuracy(model, test_dataloader)
    print("Test Accuracy: {}".format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on super graph")
    parser.add_argument("--dataset", type=str, required=True,
        help= "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit")
    parser.add_argument("--shuffle", type=str, required=True,
        help="mode of shuffle")
    parser.add_argument("--switch", type=int, default=10,
        help="number of epochs to switch shuffle mode")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=10)
    args = parser.parse_args()
    print(args)
    main(args)