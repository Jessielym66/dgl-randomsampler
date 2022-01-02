import dgl
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

from GCN_model import GCN
from sampler import MultiLayerRandomSampler

argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=1, help="GPU device ID. Use -1 for CPU training")
argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--num_layers', type=int, default=2)
argparser.add_argument('--hidden', type=int, default=256)    #16
argparser.add_argument('--fan_out', type=str, default='5,10') # 10,25
argparser.add_argument('--batch_size', type=int, default=1024)
argparser.add_argument('--val_freq', type=int, default=10)
argparser.add_argument('--lr', type=float, default=0.003)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument('--num_workers', type=int, default=1, help="Number of sampling processes. Use 0 for no extra process.")
args = argparser.parse_args()

if args.gpu >= 0:
    device = th.device('cuda:%d' % args.gpu)
else:
    device = th.device('cpu')

# load ogbn-arxiv dataset
dataset = DglNodePropPredDataset(name = args.dataset)
g, labels = dataset[0]
g = dgl.add_self_loop(g)
labels = labels[:,0]

splitted_idx = dataset.get_idx_split()
train_nid = splitted_idx['train']
val_nid = splitted_idx['valid']
test_nid = splitted_idx['test']

in_feats = g.ndata['feat'].shape[1]     # 128
n_classes = dataset.num_classes         # 40
print("in_feat:{}  n_classes:{}".format(in_feats, n_classes))

# neighborhood sample
# sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
# random sample
sampler = MultiLayerRandomSampler([int(fanout) for fanout in args.fan_out.split(',')])

dataloader = dgl.dataloading.NodeDataLoader(
    g,
    train_nid,
    sampler,
    device=device,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=args.num_workers)

# Define model and optimizer
model = GCN(in_feats, args.hidden, n_classes, args.num_layers, F.relu, args.dropout)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# writer = tensorboard.SummaryWriter(log_dir='./logs')


# set up features when using 'MultiLayerRandomSampler'
def setup_features(blocks, features, device):
    for block in blocks:
        block.srcdata['feat'] = features[block.srcdata['_ID']].to(device)
        block.dstdata['feat'] = features[block.dstdata['_ID']].to(device)
    return blocks

# Training loop
for epoch in range(1, args.epochs+1):
    model.train()   
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        # Load the input features as well as output labels
        batch_features = g.ndata['feat'][input_nodes].to(device)
        batch_labels = labels[output_nodes].to(device)
        blocks = setup_features(blocks, g.ndata['feat'], device)
        logits = model(blocks, batch_features)
        # print("blocks:", blocks[0].ndata)
        loss = F.cross_entropy(logits, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}  loss: {:.4f}'.format(epoch, loss))

    if epoch % args.val_freq == 0:
        model.eval()
        pred = model.inference(g, g.ndata['feat'], device, args.batch_size, args.num_workers)
        val_acc = (pred[val_nid].argmax(1) == labels[val_nid]).float().mean()
        test_acc = (pred[test_nid].argmax(1) == labels[test_nid]).float().mean()
        # print("pred:{} val:{} test:{}".format(pred.shape, pred[val_nid].argmax(1).shape, pred[test_nid].argmax(1).shape)) # [16w,40],[2.9w],[4.9w]
        print('Eval Acc {:.4f}'.format(val_acc))
        print('Test Acc: {:.4f}'.format(test_acc))

        #writer.add_scalar('Acc/val_acc', val_acc.cpu().detach().numpy(), global_step=epoch)
        #writer.add_scalar('Acc/test_acc', test_acc.cpu().detach().numpy(), global_step=epoch)




