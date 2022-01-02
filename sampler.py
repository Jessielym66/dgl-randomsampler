import dgl
import torch

class MultiLayerRandomSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts):
        super().__init__(len(fanouts))
        self.fanouts = fanouts

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            n_nodes = g.number_of_nodes()                 
            if isinstance(seed_nodes, dict):
                n_dst = seed_nodes['_N'].shape[0]
                seed_nodes = seed_nodes['_N']
            else:
                n_dst = seed_nodes.shape[0]
            src = torch.randint(0,n_nodes,size=(int(fanout*n_dst),))
            dst = seed_nodes.repeat(fanout)
            frontier = dgl.graph((src,dst), num_nodes=n_nodes)
        return frontier

    def __len__(self):
        return self.num_layers

        
        







