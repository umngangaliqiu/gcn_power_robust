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

torch.manual_seed(1234)
np.random.seed(1234)
# TODO: write a function to generate our adj, features, etc.
bus_no = 118
bus118data = scipy.io.loadmat('dataset_118bus.mat')
data_inputs = bus118data['input_nodal'].astype(np.float32)
data_labels = bus118data['labels'].astype(np.float32)
data_adj = bus118data['adj']


window_size = np.minimum(10, data_inputs.shape[1])
train_portion = int(0.8 * window_size)
train_x = data_inputs[:, :train_portion]
train_y = data_labels[:, :train_portion]
test_x = data_inputs[:, train_portion:]
test_y = data_labels[:, train_portion:]

input_train_mask = np.array(np.ones(bus_no), dtype=np.bool)
input_val_mask = np.array(np.zeros(bus_no), dtype=np.bool)
input_test_mask = np.array(np.ones(bus_no), dtype=np.bool)


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1* (epoch // 30))
#     for _, param_group in optimizer.state_dict():
#         param_group['lr'] = lr
#     return optimizer

def fgsm_attack(input_clean, epsilon, data_grad):
    sign_datagrad = data_grad.sign()
    purturbed_input = input_clean + epsilon * sign_datagrad
    return purturbed_input


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr  #* (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        correct = F.mse_loss(logits, labels, reduction="mean")
        return correct.item()


def evaluate_real_image(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return np.sum(np.array((logits-labels)**2))


def main(args):

    train_mask = input_train_mask
    test_mask = input_test_mask
    in_feats = args.n_input_features
    n_classes = args.n_classes
    g = DGLGraph(data_adj)
    g.add_edges(g.nodes(), g.nodes())

    n_edges = g.number_of_edges()
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0

    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.leaky_relu,
                args.dropout)

    loss_fcn = torch.nn.MSELoss()


    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer.state_dict()['param_groups'][0]['lr'])

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):

        # print("learning_rate", scheduler.get_lr())
        # adjust_learning_rate(optimizer=optimizer, epoch=epoch)

        shufle_index = np.arange(int(train_portion))
        np.random.shuffle(shufle_index)
        for t in range(train_portion):
            features, labels = load_data(shufle_index[t])
            features.requires_grad = True

            model.train()
            if epoch >= 3:
                t0 = time.time()

            # forward
            logits = model(features)
            # TO DO: change this
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()

            adjust_learning_rate(optimizer, epoch)

            loss.backward()
            data_grad = features.grad.data
            adv_feature = fgsm_attack(features, 10, data_grad)
            adv_logits = model(adv_feature)
            loss_adv = loss_fcn(adv_logits[train_mask], labels[train_mask])
            loss_adv.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(model, features, labels, train_mask)

            learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
            # print("Epoch {:05d} | learning_rate {:.4f} | Iter {:05d}|  Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} |"
            #       "ETputs(KTEPS) {:.2f}". format(epoch, learning_rate, t, np.mean(dur), loss.item(),
            #                                      acc, n_edges / np.mean(dur) / 1000))
    test(model, 10, test_mask)


def test(model, epsilon,test_mask):
    acc_temp = []
    acc_temp_attack = []
    loss_fcn_test = torch.nn.MSELoss()
    for iter in range(window_size-train_portion):
        iter = train_portion+iter
        feature_test = load_data(iter)[0]
        label_test = load_data(iter)[1]
        feature_test.requires_grad = True
        logits_test = model(feature_test)
        loss = loss_fcn_test(logits_test[test_mask], label_test[test_mask])
        model.zero_grad()
        loss.backward()
        data_grad = feature_test.grad.data
        attacked_feature = fgsm_attack(feature_test, epsilon, data_grad)
        acc_temp.append(np.array(evaluate_real_image(model, feature_test, label_test, test_mask)))
        acc_temp_attack.append(np.array(evaluate_real_image(model, attacked_feature, label_test, test_mask)))

    acc_test = np.sqrt(np.mean(acc_temp))
    acc_test_attack = np.sqrt(np.mean(acc_temp_attack))
    print("test accuracy {:.4f}| test accuracy with attack {:.4f}".format(acc_test, acc_test_attack))

# #    TODO: test
#     acc_temp = []
#     for iter in range(window_size-train_portion):
#         iter = train_portion+iter
#         acc_temp.append(np.array(evaluate_real_image(model, load_data(iter)[0], load_data(iter)[1], test_mask)))
#
#     acc_test = np.sqrt(np.mean(acc_temp))
#     print("test accuracy {:.4f}".format(acc_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.0001,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10,
            help="number of training epochs")
    parser.add_argument("--n-input-features", type=int, default=3,
            help="number of input features")
    parser.add_argument("--n-classes", type=int, default=2,
            help="number of outputs per node")
    parser.add_argument("--n-hidden", type=int, default=512,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=6,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-5,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
