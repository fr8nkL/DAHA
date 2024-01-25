# DAHA

The code for data and hardware cost model, group-based in-turn pipelined sampling and inter-batch scheduling for GNN training. Make a directory under root named ./data with three sub-directories ./data/OGB, ./data/DGL, and ./data/PYG. And the public datasets will be automatically downloaded to the corresponding directories.

To enable the intra-batch and inter-batch optimizations, first go to /src and run 
```
python HybridCPUGPU.py --dataset=<input graph name> --gnn=GraphSAGE --batch=1 --gpu=0 --runs=100 --test_mode=0
```
to retrieve the estimates and cost models. Note that for one machine, you can just use one input to retrieve the cost models and they can be applied to other input graphs on the same machine. 
To run DAHA with Group-based in-turn pipelining (including adaptive shuffling), go to /src and run 
```
python shuffle.py --dataset=<input graph name> --shuffle='off & on'
```
.
