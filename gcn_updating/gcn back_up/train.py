import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import scipy.io

from gcn import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN
# TODO: write a function to generate our adj, features, etc.
bus_no = 118
bus118data = scipy.io.loadmat('dataset_118bus.mat')
data_inputs = bus118data['input_nodal'].astype(np.float32)
data_labels = bus118data['labels'].astype(np.float32)
data_adj = bus118data['adj']


window_size = np.minimum(100000, data_inputs.shape[1])
# split data into training and testing
# split_portion = int(0.8 * data_inputs.shape[1])
train_portion = int(0.8 * window_size)
train_x = data_inputs[:, :train_portion]
train_y = data_labels[:, :train_portion]
test_x = data_inputs[:, train_portion:]
test_y = data_labels[:, train_portion:]



def load_data(iter):
    features_v = data_inputs[:bus_no]
    zz = len(data_inputs)
    features_p = data_inputs[bus_no:2*bus_no]
    features_q = data_inputs[2*bus_no: ]
    labels_mag = data_labels[0:bus_no]
    labels_ang = data_labels[bus_no:]
    features_stack = np.stack([features_v[:, iter], features_p[:, iter], features_q[:, iter]], axis=1)
    labels_ang_rad = np.dot(np.pi / 180.0, labels_ang[:, iter])
    labels_stack = np.stack([labels_mag[:, iter] * np.cos(labels_ang_rad), labels_mag[:, iter] * np.sin(labels_ang_rad)], axis=1).astype(np.float32)
    return torch.from_numpy(features_stack), torch.from_numpy(labels_stack)


input_train_mask = np.array(np.ones(bus_no), dtype=np.bool)
input_val_mask = np.array(np.zeros(bus_no), dtype=np.bool)
input_test_mask = np.array(np.ones(bus_no), dtype=np.bool)

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        correct = F.mse_loss(logits, labels, reduction="mean")
        return correct.item()

def main(args):
    # load and preprocess dataset
    # data = load_data(args)
    # features = torch.FloatTensor(data.features)
    # features = torch.from_numpy(features_stacked)
    features, labels = load_data(0)
    # labels = torch.LongTensor(data.labels)
    # labels = torch.from_numpy(label_stacked)
    train_mask = input_train_mask
    val_mask = input_val_mask
    test_mask = input_test_mask

    # if hasattr(torch, 'BoolTensor'):
    #     train_mask = torch.BoolTensor(data.train_mask)
    #     val_mask = torch.BoolTensor(data.val_mask)
    #     test_mask = torch.BoolTensor(data.test_mask)
    # else:
    #     train_mask = torch.ByteTensor(data.train_mask)
    #     val_mask = torch.ByteTensor(data.val_mask)
    #     test_mask = torch.ByteTensor(data.test_mask)
    # TODO: change following
    in_feats = args.n_input_features    #features.shape[1]
    # n_classes = data.num_labels
    n_classes = args.n_classes

    # n_edges = data.graph.number_of_edges()
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes,
    #           train_mask.int().sum().item(),
    #           val_mask.int().sum().item(),
    #           test_mask.int().sum().item()))

    # if args.gpu < 0:
    #     cuda = False
    # else:
    #     cuda = True
    #     torch.cuda.set_device(args.gpu)
    #     features = features.cuda()
    #     labels = labels.cuda()
    #     train_mask = train_mask.cuda()
    #     val_mask = val_mask.cuda()
    #     test_mask = test_mask.cuda()

    g = DGLGraph(data_adj)
    g.add_edges(g.nodes(), g.nodes())
    # graph preprocess and calculate normalization factor
    # g = data.graph
    # add self loop
    # if args.self_loop:
    #     g.remove_edges_from(nx.selfloop_edges(g))
    #     g.add_edges_from(zip(g.nodes(), g.nodes()))
    # g = DGLGraph(g)
    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    # if cuda:
    #     norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    # if cuda:
    #     model.cuda()
        # TO DO: change this
    # loss_fcn = torch.nn.CrossEntropyLoss()
    loss_fcn = torch.nn.MSELoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        shufle_index = np.arange(int(train_portion))
        np.random.shuffle(shufle_index)
        for t in range(train_portion):
            features, labels = load_data(shufle_index[t])
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            logits = model(features)
    # TO DO: change this
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(model, features, labels, train_mask)
            print("Iter {:05d}| Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(t, epoch, np.mean(dur), loss.item(),
                                                 acc, n_edges / np.mean(dur) / 1000))


# TODO: test
    for iter in range(window_size-train_portion):
        acc_test = evaluate(model, load_data(iter)[0], load_data(iter)[1], test_mask)

    print("test accuracy {:.4f}".format(np.sqrt(acc_test)))

    # acc = evaluate(model, features, labels, test_mask)
    # print("Test accuracy {:.2%} | output ".format(acc), torch.cat([logits, labels], dim=1))


if __name__ == '__main__':
    torch.manual_seed(1234)
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-4,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("--n-input-features", type=int, default=3,
            help="number of input features")
    parser.add_argument("--n-classes", type=int, default=2,
            help="number of outputs per node")
    parser.add_argument("--n-hidden", type=int, default=256,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-5,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    args.dataset = "cora"
    print(args)

    main(args)
