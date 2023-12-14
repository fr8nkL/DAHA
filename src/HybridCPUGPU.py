#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import time
from matplotlib import pyplot as plt
import tqdm
import sklearn.metrics
from statistics import median
import math

import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from torch_geometric.datasets import Coauthor, Amazon
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from dgl.nn.pytorch.conv import GraphConv

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from DAHA import hybrid_sample

OGB_DIR = "../data/OGB"
DGL_DIR = "../data/DGL"
PYG_DIR = "../data/PYG"

def add_file_postfix(directory, s):
    existing_files = os.listdir(directory)
    flag = s in existing_files
    name = s
    if flag:
        splits = s.split('.')
        break_idx = len(s) - len(splits[-1]) - 1
        postfix = 0
        flag = True
        while flag:
            name = "{}_{}.{}".format(s[:break_idx], postfix, splits[-1])
            flag = name in existing_files
            postfix += 1
            # print(name, flag)
    return "{}/{}".format(directory, name)

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
    print(graph)
    return graph, train_nids, valid_nids, test_nids

class Model(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(Model, self).__init__()
        # self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        # self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.conv = SAGEConv(in_feats, num_classes, aggregator_type='gcn')
        # self.h_feats = h_feats

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        # h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        # h = self.conv1(mfgs[0], (x, h_dst))  # <---
        # h = F.relu(h)
        # h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
        # h = self.conv(mfgs[-1], (h, h_dst))  # <---
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv(mfgs[-1], (x, h_dst))
        return h
    
class ModelFull(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(ModelFull, self).__init__()
        # self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        # self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.conv = SAGEConv(in_feats, num_classes, aggregator_type='gcn')
        # self.h_feats = h_feats

    def forward(self, graph, x):
        # Lines that are changed are marked with an arrow: "<---"

        # h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
        # h = self.conv1(mfgs[0], (x, h_dst))  # <---
        # h = F.relu(h)
        # h_dst = h[:mfgs[-1].num_dst_nodes()]  # <---
        # h = self.conv(mfgs[-1], (h, h_dst))  # <---
        h_dst = x
        h = self.conv(graph, (x, h_dst))
        return h

class GCNNet(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GCNNet, self).__init__()
        self.conv = GraphConv(in_feats, num_classes, activation=torch.nn.Sigmoid(), allow_zero_in_degree=True)
    def forward(self, graph):
        h = graph.ndata['feat']
        h = self.conv(graph, h)
        return h

def test_dgl_full(graph, num_features, num_classes, device, num_epochs, warmups=0):
    model = ModelFull(num_features, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters())
    graph = graph.to(device)
    for epoch in tqdm.tqdm(range(num_epochs)):
        if epoch == warmups:
            torch.cuda.synchronize()
            t0 = time.time()
        model.train()
        # feature copy from CPU to GPU takes place here
        # inputs = mfgs[0].srcdata['feat']
        inputs = graph.ndata['feat']
        labels = graph.ndata['label']

        predictions = model(graph, inputs)

        loss = F.cross_entropy(predictions, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / (num_epochs - warmups)

def test_dgl(train_dataloader_cpu_no_shuffle, num_features, num_classes, device, num_epochs, warmups=0):
    # GraphSAGE
    model = Model(num_features, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters())
    durs = []
    for epoch in tqdm.tqdm(range(num_epochs)):
        durs_batch = []
        if epoch == warmups:
            torch.cuda.synchronize()
            t0 = time.time()
        model.train()
        # for _, _, mfgs in tqdm.tqdm(train_dataloader_cpu_no_shuffle, leave=False):
        for _,_,mfgs in train_dataloader_cpu_no_shuffle:
            torch.cuda.synchronize()
            t2 = time.time()
            mfgs[-1] = mfgs[-1].to(device)
            # feature copy from CPU to GPU takes place here
            # inputs = mfgs[0].srcdata['feat']
            inputs = mfgs[-1].srcdata['feat']
            labels = mfgs[-1].dstdata['label']
            torch.cuda.synchronize()
            t3 = time.time()
            predictions = model(mfgs, inputs)
            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            t4 = time.time()
            durs_batch.append([t3 - t2, t4 - t3]) # [[data move, train]]
        durs.append(durs_batch)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / (num_epochs - warmups), durs

def test_dgl_saint(train_dataloader_cpu_no_shuffle, num_features, num_classes, device, num_epochs, warmups=0):
    # GraphSAINT with GCN
    model = GCNNet(num_features, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters())
    durs = []
    for epoch in tqdm.tqdm(range(num_epochs)):
        durs_batch = []
        if epoch == warmups:
            torch.cuda.synchronize()
            t0 = time.time()
        model.train()
        # for _, _, mfgs in tqdm.tqdm(train_dataloader_cpu_no_shuffle, leave=False):
        for g in train_dataloader_cpu_no_shuffle:
            torch.cuda.synchronize()
            t2 = time.time()
            g = g.to(device)
            inputs = g.ndata['feat']
            labels = g.ndata['label']
            torch.cuda.synchronize()
            t3 = time.time()
            predictions = model(g)
            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            t4 = time.time()
            durs_batch.append([t3 - t2, t4 - t3]) # [[data move, train]]
        durs.append(durs_batch)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / (num_epochs - warmups), durs

def test_sample(dataloader,runs=10):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in tqdm.tqdm(range(runs)):
        for _ in dataloader:
            # dummy
            a=1
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / runs

def test_op(f, x1, x2, runs=10, display=False):
    durs = []
    for i in range(runs):
        t0 = time.time()
        f(x1,x2)
        t1 = time.time()
        durs.append(t1-t0)
    if display:
        plt.scatter(np.arange(runs), durs)
        plt.show()
    return durs

def test_op_gpu(f, x1, x2, runs=10, display=False, warmup=True, warmups=3, warmup_dim=10, device=torch.device('cuda')):
    if warmup:
        # perform warmups
        for _ in range(warmups):
            f(x1.to('cuda'), torch.rand((x1.size(1), warmup_dim), device='cuda'))
            torch.matmul(x2.to('cuda'), torch.rand((x2.size(1), warmup_dim), device='cuda'))
            torch.cuda.empty_cache()
    # test runs
    durs = []
    y1 = x1.to('cuda')
    y2 = x2.to('cuda')
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        f(y1,y2)
        torch.cuda.synchronize()
        t1 = time.time()
        durs.append(t1-t0)
        torch.cuda.empty_cache()
    del y1
    del y2
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if display:
        plt.scatter(np.arange(runs), durs)
        plt.show()
    return durs

def test_comm(X, runs=10, display=False, warmup=True, warmups=3):
    if warmup:
        for _ in range(warmups):
            torch.rand(1).to('cuda')
            torch.cuda.empty_cache()
    durs = []
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        Y = X.to('cuda')
        torch.cuda.synchronize()
        t1 = time.time()
        durs.append(t1-t0)
        del Y
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if display:
        plt.scatter(np.arange(runs), durs)
        plt.show()
    return durs

def test_grad(A, H, W, labels=None, runs=30, display=False, warmup=True, warmups=3, device=torch.device('cuda')):
    A = A.to(device)
    H = H.to(device)
    W = W.to(device)
    labels = labels.to(device)

    # HW = torch.mm(H,W)
    # Z = torch.sparse.mm(torch.t(A), HW)
    T = torch.sparse.mm(torch.t(A), H)
    Z = torch.mm(T, W)
    
    H_ = torch.nn.Sigmoid()(Z)
    derivative_sigmoid = lambda x: torch.nn.Sigmoid()(x) * (1 - torch.nn.Sigmoid()(x))
    H_.requires_grad_()
    # test runs
    durs = []
    for i in range(runs):
        torch.cuda.synchronize()
        
        t0 = time.time()

        loss = torch.nn.functional.cross_entropy(H_, labels)
        loss.backward()
        G = torch.mul(H_.grad, derivative_sigmoid(Z))
        torch.matmul(torch.t(T), G)

        torch.cuda.synchronize()
        t1 = time.time()
        durs.append(t1-t0)
        
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if display:
        plt.scatter(np.arange(runs), durs)
        plt.show()
    return durs

def test_batch(A, H, W, labels=None, runs=10, display=False, warmup=True, warmups=3, device=torch.device('cuda')):
    HW = torch.mm(H,W)
    mm_gpu   = test_op_gpu(torch.mm, H, W, runs=runs, display=display, warmup=warmup, warmups=warmups, device=device)
    mm_cpu   = test_op(    torch.mm, H, W, runs=runs, display=display)
    spmm_cpu = test_op(    torch.sparse.mm, torch.t(A), H, runs=runs, display=display)
    spmm_gpu = test_op_gpu(torch.sparse.mm, torch.t(A), H, runs=runs, display=display, warmup=warmup, warmups=warmups, device=device)
    spmm_gpu_AHW = test_op_gpu(torch.sparse.mm, torch.t(A), HW, runs=runs, display=display, warmup=warmup, warmups=warmups, device=device)
    grad = test_grad(A, H, W, labels, runs=runs, display=display, warmup=warmup, warmups=warmups, device=device)
    comm_A  = test_comm(A, runs=runs, display=display, warmup=warmup, warmups=warmups)
    comm_H  = test_comm(H, runs=runs, display=display, warmup=warmup, warmups=warmups)
    comm_HW = test_comm(HW, runs=runs, display=display, warmup=warmup, warmups=warmups)
    comm_W  = test_comm(W, runs=runs, display=display, warmup=warmup, warmups=warmups)
    # print(median(comm_A) / median(mm_cpu), np.mean(comm_A) / np.mean(mm_cpu))
    return [comm_A, comm_H, comm_HW, comm_W,
            spmm_gpu, spmm_cpu, mm_gpu, mm_cpu, spmm_gpu_AHW, grad]

def test_stats(data, f=np.mean):
    """
    input data is batch stats in the form of 
    [[comm_A, comm_H, comm_HW, comm_W,
    spmm_gpu, spmm_cpu, mm_gpu, mm_cpu]]
    """
    sums = []
    # batch_idx = 0
    for batch_idx in range(len(data)):
        stat = data[batch_idx]
        comm_A       = f(stat[0])
        comm_H       = f(stat[1])
        comm_HW      = f(stat[2])
        comm_W       = f(stat[3])
        spmm_gpu     = f(stat[4])
        spmm_cpu     = f(stat[5])
        mm_gpu       = f(stat[6])
        mm_cpu       = f(stat[7])
        spmm_gpu_AHW = f(stat[8])
        grad         = f(stat[9])
        # get estimated time
        time_pre_tran = max(comm_A, mm_cpu) + comm_HW + spmm_gpu_AHW + grad
        time_gpu      = comm_A + comm_H + spmm_gpu + mm_gpu + grad
        # print(batch_idx, '\t', 1 - time_pre_tran/time_gpu)
        sums.append([time_pre_tran,time_gpu])
    sums = np.array(sums)
    print("total speedup: {:.2f}%".format(100 * (1 - np.sum(sums[:,0]) / np.sum(sums[:,1]))))
    return sums

_cache_ = {
  'memory_allocated': 0,
  'max_memory_allocated': 0,
  'memory_reserved': 0,
  'max_memory_reserved': 0,
}

def _get_memory_info(info_name, unit):

    tab = '\t'
    if info_name == 'memory_allocated':
        current_value = torch.cuda.memory.memory_allocated()
    elif info_name == 'max_memory_allocated':
        current_value = torch.cuda.memory.max_memory_allocated()
    elif info_name == 'memory_reserved':
        tab = '\t\t'
        current_value = torch.cuda.memory.memory_reserved()
    elif info_name == 'max_memory_reserved':
        current_value = torch.cuda.memory.max_memory_reserved()
    else:
        raise ValueError()

    divisor = 1
    if unit.lower() == 'kb':
        divisor = 1024
    elif unit.lower() == 'mb':
        divisor = 1024*1024
    elif unit.lower() == 'gb':
        divisor = 1024*1024*1024
    else:
        raise ValueError()

    diff_value = current_value - _cache_[info_name]
    _cache_[info_name] = current_value

    return f"{info_name}: \t {current_value} ({current_value/divisor:.3f} {unit.upper()})" \
            f"\t diff_{info_name}: {diff_value} ({diff_value/divisor:.3f} {unit.upper()})"

def print_memory_info(unit='mb'):

    print(_get_memory_info('memory_allocated', unit))
    print(_get_memory_info('max_memory_allocated', unit))
    print(_get_memory_info('memory_reserved', unit))
    print(_get_memory_info('max_memory_reserved', unit))
    print('')

def ceiling(x, factor=512):
    return math.ceil(x/factor)*factor

def main(args):
    # SEED = 24
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # np.random.seed(SEED)

    if args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    print(device)

    graph, train_nids, valid_nids, test_nids = load_data(args.dataset)
    num_features = graph.ndata['feat'].shape[1]
    num_classes = (graph.ndata['label'].max() + 1).item()
    # print("##### ##### ##### ##### #####")
    print('Number of train_nodes & features & classes: {}, {}, {}'.format(len(train_nids), num_features, num_classes))

    print_memory_info()
    A = graph.adj().to(device)
    print_memory_info()
    H = graph.ndata['feat'].to(device)
    print_memory_info()
    # Y = graph.ndata['label'].to(device)
    # train_nids = train_nids.to(device)
    # valid_nids = valid_nids.to(device)
    # test_nids = test_nids.to(device)
    graph=graph.to(device)
    print_memory_info()
    # print(A)
    # print(H)

    # # before
    # # torch._C._cuda_clearCublasWorkspaces()
    # memory_before = torch.cuda.memory_allocated(device)

    # # your tensor
    # A = graph.adj().to(device)
    # H = graph.ndata['feat'].to(device)
    # # graph = graph.to(device)
    # print(graph)

    # # after
    # memory_after = torch.cuda.memory_allocated(device)
    # latent_size = memory_after - memory_before

    # print(latent_size)

    # # test 1st comm overhead
    # durs = test_comm(graph.ndata['feat'], runs=30, display=False, warmup=False, warmups=0)
    # print(durs)
    # with open('tmp.npy', 'wb') as f:
    #     np.save(f, np.array(durs))
    # return 0
    # # test 1st comm overhead

    # TODO: parameters
    # main test
    runs = args.runs
    display = False
    warmup = False
    warmups = int(0.1 * runs)

    # dense feature
    
    measure = median#np.mean#np.sum#
    n_max = graph.num_nodes() # H.size(0)
    f = num_features
    h =  num_classes#128#
    data = []

    # mm
    t0 = time.time()
    for i in tqdm.tqdm(range(20)):
        # for _ in range(3):
        n = int(np.random.rand() * n_max)
    # for f in [64, 128, 512, 1024]:
        W = torch.rand((f, h))
        y = measure(test_op_gpu(torch.mm, torch.rand((n, f)), W, runs=runs, display=False))
        # y = measure(test_op(torch.mm, torch.rand((n, f)), W, runs=runs, display=False))
        data.append([n,y])
    t1 = time.time()
    print("data collection takes {:.4f} sec".format(t1-t0))

    # spmm
    # t0 = time.time()
    # for i in tqdm.tqdm(range(10)):
    #     # for _ in range(3):
    #     n = int(np.random.rand() * n_max)
    # # for f in [64, 128, 512, 1024]:
    #     g = graph.subgraph(np.random.choice(n_max, n))
    #     y = measure(test_op_gpu(torch.spmm, g.adj(), g.ndata['feat'], runs=runs, display=False))
    #     # y = measure(test_op(torch.spmm, g.adj(), g.ndata['feat'], runs=runs, display=False))
    #     data.append([g.num_nodes(), g.num_edges(),y])
    # t1 = time.time()
    # print("data collection takes {:.4f} sec".format(t1-t0))

    # # dense feature
    # n_max = 100000 # H.size(0)
    # f = num_features # num_classes
    # data = []
    # t0 = time.time()
    # for i in tqdm.tqdm(range(10)):
    #     n = int(np.random.rand() * n_max * (i+1)/10)
    #     # for f in [64, 128, 256, 512]:
    #     y = measure(test_comm(torch.rand((n, f)), runs=runs, display=False, warmup=warmup, warmups=warmups))
    #     data.append([n,f,y])
    # t1 = time.time()
    # print("data collection takes {:.4f} sec".format(t1-t0))

    # # sparse matrix
    # t0 = time.time()
    # for i in tqdm.tqdm(range(10)):
    #     n = int(np.random.rand() * n_max * 5)
    #     g = graph.subgraph(np.random.choice(n_max, n))
    #     y = measure(test_comm(g.adj(), runs=runs, display=False))
    #     data.append([g.num_nodes(), g.num_edges(),y])
    # t1 = time.time()
    # print("data collection takes {:.4f} sec".format(t1-t0))
    
    # with open('tmp.npy'.format(h), 'wb') as f:
    #     np.save(f, np.array(data))

    # with open('tmp.npy'.format(h), 'rb') as f:
    #     data = np.load(f)

    data = np.array(data)
    # X = data[:,[1]]
    X = data[:,:-1]
    y = data[:,-1]
    # X, y
    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(X)
    t0 = time.time()
    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    t1 = time.time()
    print("regression takes {:.4f} sec".format(t1-t0))
    print(data)
    lin2.predict(poly.fit_transform(X))

    # plt.scatter(range(len(X)), y, color='blue')
 
    # plt.plot(range(len(X)), lin2.predict(poly.fit_transform(X)),
    #         color='red')
    # plt.title('Polynomial Regression')
    # plt.xlabel('Temperature')
    # plt.ylabel('Pressure')
    
    # plt.show()

    A = graph.adj()
    H = graph.ndata['feat']
    W = torch.rand((num_features, h))
    
    print(measure(test_op_gpu(torch.mm, H, W, runs=runs, display=False)))
    print(lin2.predict(poly.fit_transform([[H.size(0)]]))[0])
    t0 = time.time()
    for _ in range(100):
        lin2.predict(poly.fit_transform([[H.size(0)]]))[0]
    t1 = time.time()
    print(t1-t0)

    # print(measure(test_op_gpu(torch.spmm, A, H, runs=runs, display=False)))
    # print(lin2.predict(poly.fit_transform([[graph.num_nodes(), graph.num_edges()]]))[0])

    # print(measure(test_comm(H, runs=runs, display=True)))
    # print(lin2.predict(poly.fit_transform([[graph.num_nodes(), num_features]]))[0])

    # print(measure(test_comm(A, runs=runs, display=True)))
    # print(lin2.predict(poly.fit_transform([[graph.num_nodes(), graph.num_edges()]]))[0])

    
    # model param
    gnn = args.gnn
    num_sampler = args.num_sampler
    batch_size = args.batch_size
    budget = args.budget
    fanouts = [5] # [5, 10]
    drop_last = True

    # output param
    output_dir = '../output/HybridPU' # no slash at the end

    num_batch = int(len(train_nids) / batch_size) if drop_last == True else int(np.ceil(len(train_nids) / batch_size))
    if args.batch != 0:
        print('number of batches per epoch: {}'.format(num_batch))

    # MAIN
    batch_stats = []
    if args.batch == 0:
        print("full-graph training")
        print("Measuring DGL")
        dgl_time = test_dgl_full(graph, num_features, num_classes, device, runs, warmups)
        print("Measuring hybridPU")
        batch_stats.append(test_batch(
            graph.adj(), graph.ndata['feat'], 
            torch.rand((num_features, num_classes)), graph.ndata['label'],
            runs=runs, display=display, warmup=warmup, warmups=warmups, device=device
        ))
        cpu_sample_time = 0.
    else:
        print("mini-batch training")
        # test time cost of transfer the entire adj matrix
        print("Measuring transfer cost of the entire matrix")
        comm_A_full_mean = test_comm(graph.adj().to('cpu'), runs=runs, display=display, warmup=warmup, warmups=warmups)
        comm_A_full_mean = np.mean(comm_A_full_mean)
        if gnn == 'GraphSAGE':
            print("Measuring DGL")
            sampler = dgl.dataloading.NeighborSampler(fanouts)
            train_dataloader_cpu_no_shuffle = dgl.dataloading.DataLoader(
                # The following arguments are specific to DGL's DataLoader.
                graph,              # The graph
                train_nids,         # The node IDs to iterate over in minibatches
                sampler,            # The neighbor sampler
                device=torch.device('cpu'),      # Put the sampled MFGs on CPU or GPU
                # The following arguments are inherited from PyTorch DataLoader.
                batch_size=batch_size,    # Batch size
                shuffle=False,       # Whether to shuffle the nodes for every epoch
                drop_last=drop_last,    # Whether to drop the last incomplete batch
                num_workers=num_sampler       # Number of sampler processes
            )
            # dgl_time, dgl_durs = test_dgl(train_dataloader_cpu_no_shuffle, graph, train_nids, num_features, num_classes, device, runs, batch_size, fanouts, warmups, num_sampler)
            dgl_time, dgl_durs = test_dgl(train_dataloader_cpu_no_shuffle, num_features, num_classes, device, runs, warmups)
            print("Measuring hybridPU")
            for _, _, mfgs in tqdm.tqdm(train_dataloader_cpu_no_shuffle):
                batch_stats.append(test_batch(
                    mfgs[-1].adj(), mfgs[-1].srcdata['feat'], 
                    torch.rand((num_features, num_classes)), mfgs[-1].dstdata['label'],
                    runs=runs, display=display, warmup=warmup, warmups=warmups, device=device
                ))
        elif gnn == 'GraphSAINT':
            sampler = dgl.dataloading.SAINTSampler(mode='node', budget=budget)
            train_dataloader_cpu_no_shuffle = dgl.dataloading.DataLoader(
                graph, 
                torch.arange(num_batch), 
                sampler, 
                device=torch.device('cpu'),
                shuffle=False,
                drop_last=drop_last,
                num_workers=0
            )
            print("Measuring DGL")
            dgl_time, dgl_durs = test_dgl_saint(train_dataloader_cpu_no_shuffle, num_features, num_classes, device, runs, warmups)
            print("Measuring hybridPU")
            for g in tqdm.tqdm(train_dataloader_cpu_no_shuffle):
                batch_stats.append(test_batch(
                    g.adj(), g.ndata['feat'], 
                    torch.rand((num_features, num_classes)), g.ndata['label'],
                    runs=runs, display=display, warmup=warmup, warmups=warmups, device=device
                ))
        else:
            raise NotImplementedError
        # measure sample time
        print("Measuring CPU sampling time")
        cpu_sample_time = test_sample(train_dataloader_cpu_no_shuffle, runs=runs)
        print("Measuring GPU sampling time")
        if gnn == 'GraphSAGE':
            sampler = dgl.dataloading.NeighborSampler(fanouts)
            train_dataloader_gpu_no_shuffle = dgl.dataloading.DataLoader(
                # The following arguments are specific to DGL's DataLoader.
                graph.to(device),              # The graph
                train_nids.to(device),         # The node IDs to iterate over in minibatches
                sampler,            # The neighbor sampler
                device=device,      # Put the sampled MFGs on CPU or GPU
                # The following arguments are inherited from PyTorch DataLoader.
                batch_size=batch_size,    # Batch size
                shuffle=False,      # Whether to shuffle the nodes for every epoch
                drop_last=drop_last,    # Whether to drop the last incomplete batch
                num_workers=0,      # Number of sampler processes
                use_uva=False
            )
        elif gnn == "GraphSAINT":
            sampler = dgl.dataloading.SAINTSampler(mode='node', budget=budget)
            train_dataloader_gpu_no_shuffle = dgl.dataloading.DataLoader(
                graph.to(device), 
                torch.arange(num_batch, device=device), 
                sampler, 
                device=device,
                shuffle=False,
                drop_last=drop_last,
                num_workers=0,
                use_uva=False
            )
        else:
            raise NotImplementedError
        gpu_sample_time = test_sample(train_dataloader_gpu_no_shuffle, runs=runs)
    
    # output stats
    sums = test_stats(batch_stats, np.mean)
    pre_tran_time = np.sum(sums[:,0]) # per-epoch
    # print("breakdown:\n{}\n{}\n{}".format(*np.array([pre_tran_time, cpu_sample_time, dgl_time]) * runs))
    # below are all per epoch
    print("classic_time     = {}".format(np.sum(sums[:,1])))
    print("pre_tran_time    = {}".format(pre_tran_time))
    print("dgl              = {}".format(dgl_time))
    print("intra            = {}".format(pre_tran_time + cpu_sample_time))
    print("cpu    sample + intra     vs dgl    : {:.2f}%".format(100 * (1 - (pre_tran_time + cpu_sample_time) / dgl_time)))
    if args.batch != 0:
        # stats = np.array(dgl_durs)
        # num_batch = stats.shape[1]
        # print("number of batch: {}".format(num_batch))
        
        # "estimate_salient()" returns total time not per epoch time
        # salient_per_batch_stat = np.mean(np.mean(np.array(dgl_durs)[warmups:,:,:], axis=0)[:-1], axis=0)
        # salient_total = estimate_salient(cpu_sample_time, salient_per_batch_stat[0], salient_per_batch_stat[1], num_batch, runs)

        # salient with cpu sampling speed
        salient_per_batch = [[cpu_sample_time/num_batch, *np.mean(np.array(dgl_durs)[:,i,:], axis=0)] for i in range(num_batch)]

        # salient with gpu sampling speed
        salient_per_batch = [[gpu_sample_time/num_batch, *np.mean(np.array(dgl_durs)[:,i,:], axis=0)] for i in range(num_batch)]

        # transfer cost of batch adj for one epoch
        comm_dA_epoch_mean = np.mean(np.array(batch_stats)[:,0,:]) * num_batch
        # hybrid sample time per epoch
        # the hybrid sample stage ends when GPU yields x epochs
        # hence, with larger x, better efficiency
        # Also, adj can be stored in GPU for the entire training process given enough GPU memory,
        # which means x can be number of epochs.
        hybrid_sample_time = hybrid_sample(train_dataloader_cpu_no_shuffle, train_dataloader_gpu_no_shuffle, comm_A_full_mean, comm_dA_epoch_mean, x=args.gpu_s_ratio)
        # inter-batch pipelined with intra-opt 
        inter_per_epoch_time = hybrid_inter(batch_stats)
        # hybrid sample + intra opt time per epoch
        hybrid_intra = hybrid_sample_time + pre_tran_time
        # hybrid sample + inter opt time per epoch
        hybrid_inter = hybrid_sample_time + inter_per_epoch_time
        # # estimate hybrid sample time without considering the transfer cost of adjs
        # def estimate_hybrid(cpu_sample_per_epoch, gpu_sample_per_epoch, pre_tran_per_epoch):
        #     return (cpu_sample_per_epoch / (1 + cpu_sample_per_epoch / gpu_sample_per_epoch) + pre_tran_per_epoch)
        # hybrid_intra = estimate_hybrid(cpu_sample_time, gpu_sample_time, pre_tran_time)
        # hybrid_inter = estimate_hybrid(cpu_sample_time, gpu_sample_time, inter_per_epoch_time)
        # # print("intra vs dgl: {:.2f}%".format(100 * (1 - (pre_tran_time + cpu_sample_time) / dgl_time)))
        print("hybrid_intra     = {}".format(hybrid_intra))
        print("hybrid_inter     = {}".format(hybrid_inter))
        print("inter_opt        = {}".format(inter_per_epoch_time))
        print("comm_A_full_mean = {}".format(comm_A_full_mean))
        print("hybrid_sample    = {}".format(hybrid_sample_time))
        print("gpu_sample_time  = {}".format(gpu_sample_time))
        print("cpu_sample_time  = {}".format(cpu_sample_time))
    # save to file
    if args.test_mode == 0:
        output_path = add_file_postfix(output_dir, "op_breakdown_{}.npy".format(args.dataset))
        with open(output_path, 'wb') as f:
            np.save(f, batch_stats)
        output_path_dgl = add_file_postfix(output_dir, "dgl_breakdown_{}.npy".format(args.dataset))
        with open(output_path_dgl, 'wb') as f:
            if args.batch != 0:
                np.save(f, np.array([dgl_time, dgl_durs], dtype=object))
        print("saved to {} and {}".format(output_path, output_path_dgl))
    if args.batch != 0:
        assert num_batch == np.array(batch_stats).shape[0]
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on super graph")
    parser.add_argument("--dataset", type=str, required=True,
        help= "The input dataset. Can be cora, citeseer, pubmed, syn(synthetic dataset) or reddit")
    parser.add_argument("--gnn", type=str, default="GraphSAGE")
    parser.add_argument("--batch", type=int, default=0,
                        help="full-graph training if 0 else mini-batch training")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--budget", type=int, default=16384)
    parser.add_argument("--num_sampler", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--gpu_s_ratio", type=int, default=50)
    parser.add_argument("--test_mode", type=int, default=0,
                        help="if 0, just test no output")
    
    args = parser.parse_args()
    print(args)
    main(args)