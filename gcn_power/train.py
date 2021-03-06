import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args
import scipy.io
from torch.autograd import Variable
import matplotlib.pyplot as plt


from gcn import GCN
import random

def init_seed(seed=123):
    '''set seed of random number generators'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


# torch.manual_seed(0)
# np.random.seed(0)
# TODO: write a function to generate our adj, features, etc.
bus_no = 118
bus118data = scipy.io.loadmat('dataset_118bus.mat')
data_inputs = bus118data['input_nodal'].astype(np.float32)
data_labels = bus118data['labels'].astype(np.float32)
data_adj = bus118data['adj']

init_seed()
window_size = np.minimum(20, data_inputs.shape[1])
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

def ifgsm_attack(model, loss_fcn, optimizer, feature, label, T_adv, epsilon, lr):
    feature, label = Variable(feature), Variable(label)
    adv_feature = feature.data.clone()
    adv_feature = Variable(adv_feature, requires_grad=True)

    for n in range(T_adv):  # Q: why we need pass optimizer?
        optimizer.zero_grad()
        loss_adv = loss_fcn(model(adv_feature), label)
        loss_adv.backward()
        data_grad = adv_feature.grad.data

        grad_ = data_grad.view(len(adv_feature), -1)
        grad_ = grad_/torch.norm(grad_,2,1).view(len(adv_feature),1).expand_as(grad_)
        delta_x = epsilon * grad_.view_as(data_grad)
        delta_x[delta_x!=delta_x]=0
        delta_x.clamp_(-epsilon, epsilon)
        step_adv = adv_feature.data + lr * delta_x
        total_adv = step_adv - feature.data
        total_adv.clamp_(-epsilon, epsilon)
        adv_feature.data = feature.data + total_adv

    return adv_feature

def fgsm_attack_generateor(model, loss_fcn, feature, label, epsilon_fgsm, mask):
    feature, label = Variable(feature), Variable(label)
    adv_feature = feature.data.clone()
    adv_feature = Variable(adv_feature, requires_grad=True)
    logits = model(adv_feature)
    loss = loss_fcn(logits[mask], label[mask])
    model.zero_grad()
    loss.backward()
    data_grad = adv_feature.grad.data
    sign_datagrad = data_grad.sign()
    adv_feature = feature + epsilon_fgsm * sign_datagrad
    return adv_feature

def fgsm_attack(input_clean, epsilon, data_grad):
    sign_datagrad = data_grad.sign()
    purturbed_input = input_clean + epsilon * sign_datagrad
    return purturbed_input

def distr_attack(model, loss_fcn, feature_train, label_train, gamma, T_adv):
    feature_train, label_train = Variable(feature_train), Variable(label_train)
    adv_feature = feature_train.data.clone()
    adv_feature = Variable(adv_feature, requires_grad=True)

    # Running maximizer for adversarial example
    optimizer_adv = torch.optim.Adam([adv_feature], lr=0.2)
    loss_phi = 0  # phi(theta,z0)
    rho = 0  # E[c(Z,Z0)]
    for n in range(T_adv):
        optimizer_adv.zero_grad()
        delta = adv_feature - feature_train
        rho = torch.mean((torch.norm(delta.view(len(feature_train), -1), 2, 1) ** 2))
        # rho = torch.mean((torch.norm(z_hat-x_,2,1)**2))
        loss_zt = loss_fcn(model(adv_feature), label_train)
        # -phi_gamma(theta,z)
        loss_phi = - (loss_zt - gamma * rho)
        loss_phi.backward()
        optimizer_adv.step()
    return adv_feature
        # adjust_lr_zt(optimizer_zt, max_lr0, n + 1)


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
    model_adv = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.leaky_relu,
                args.dropout)

    model_non_adv = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.leaky_relu,
                args.dropout)

    loss_fcn = torch.nn.MSELoss()


    # use optimizer
    optimizer = torch.optim.Adam(model_adv.parameters(),
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
            # features.requires_grad = True

            model_non_adv.train()
            model_adv.train()
            # if epoch >= 3:
            t0 = time.time()
            # adv_features = distr_attack(model=model, loss_fcn=loss_fcn, feature_train=features,
            #                             label_train=labels, gamma=0.000001, T_adv=20)

            # adv_features = ifgsm_attack(model=model, loss_fcn=loss_fcn, optimizer=optimizer, feature=features,
            #                             label=labels, T_adv=20, epsilon=1, lr=0.1)

            adv_features = fgsm_attack_generateor(model=model_adv, loss_fcn=loss_fcn, feature=features, label=labels,
                                                epsilon_fgsm=0, mask=train_mask)

            # forward
            logits = model_adv(adv_features)
            # TO DO: change this
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()

            adjust_learning_rate(optimizer, epoch)

            loss.backward()

            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc = evaluate(model_adv, features, labels, train_mask)

            learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
            print("Epoch {:05d} | learning_rate {:.4f} | Iter {:05d}|  Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} |"
                  "ETputs(KTEPS) {:.2f}". format(epoch, learning_rate, t, np.mean(dur), loss.item(),
                                                 acc, n_edges / np.mean(dur) / 1000))

    history_acc_test = {}
    history_acc_test["nominal"] = []
    # history_acc_test["distr"] = []
    history_acc_test["fgsm"] = []
    history_acc_test["ifgsm"] = []

    # eps = [0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08]
    eps = [0, .001, .005, .009, .02, 0.05, 0.08]
    for _, i_eps in enumerate(eps):
        # history_acc_test["distr"].append(test_distr(model_adv, loss_fcn, gamma=0.001, T_adv=20, test_mask=test_mask))
    # Increasing T_adv does not affect very much, smaller gamma and smaller T_adv the better
        out = test_fgsm(model_adv, epsilon=i_eps, test_mask=test_mask)
        history_acc_test["nominal"].append(out[0])
        history_acc_test["fgsm"].append(out[1])
        history_acc_test["ifgsm"].append(test_ifgsm(model_adv, loss_fcn, optimizer, epsilon=i_eps, T_adv=20, lr=0.1, test_mask=test_mask))

    plot_graphs(data=history_acc_test)

def test_distr(model, loss_fcn, gamma, T_adv, test_mask):
    model.eval()
    acc_temp = []
    loss_fcn_test = torch.nn.MSELoss()
    for iter in range(train_portion, window_size):
        feature_test = load_data(iter)[0]
        label_test = load_data(iter)[1]

        feature_test, label_test = Variable(feature_test), Variable(label_test)
        adv_feature = feature_test.data.clone()
        adv_feature = Variable(adv_feature, requires_grad=True)

        # Running maximizer for adversarial example
        optimizer_adv = torch.optim.Adam([adv_feature], lr=2)
        loss_phi = 0  # phi(theta,z0)
        rho = 0  # E[c(Z,Z0)]
        for n in range(T_adv):
            optimizer_adv.zero_grad()
            delta = adv_feature - feature_test
            rho = torch.mean((torch.norm(delta.view(len(feature_test), -1), 2, 1) ** 2))
            # rho = torch.mean((torch.norm(z_hat-x_,2,1)**2))
            loss_zt = loss_fcn(model(adv_feature), label_test)
            # -phi_gamma(theta,z)
            loss_phi = - (loss_zt - gamma * rho)
            loss_phi.backward()
            optimizer_adv.step()
            # adjust_lr_zt(optimizer_zt, max_lr0, n + 1)

        acc_temp.append(np.array(evaluate_real_image(model, adv_feature, label_test, test_mask)))

    acc_test = np.sqrt(np.mean(acc_temp))
    print("test accuracy with distributional attack {:.4f}".format(acc_test))
    return acc_test


def test_fgsm(model, epsilon, test_mask):
    model.eval()
    acc_temp = []
    acc_temp_attack = []
    loss_fcn_test = torch.nn.MSELoss()
    for iter in range(train_portion, window_size):
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
    print("test accuracy {:.4f}| test accuracy with FGSM attack {:.4f}".format(acc_test, acc_test_attack))
    return acc_test, acc_test_attack


def test_ifgsm(model, loss_fcn, optimizer, epsilon, T_adv, lr, test_mask):
    model.eval()
    acc_temp_ifgsm = []

    for iter in range(train_portion, window_size):
        feature_test = load_data(iter)[0]
        label_test = load_data(iter)[1]
        adv_feature = ifgsm_attack(model, loss_fcn, optimizer, feature_test, label_test, T_adv, epsilon, lr)
        acc_temp_ifgsm.append(np.array(evaluate_real_image(model, adv_feature, label_test, test_mask)))
    acc_test_ifgm = np.sqrt(np.mean(acc_temp_ifgsm))
    print("test accuracy with iFGSM attack {:.4f}".format(acc_test_ifgm))
    return acc_test_ifgm

def plot_graphs(data):
    fig = plt.figure(figsize=(5, 5))
    Colors = ['blue', 'orange', 'red', 'purple', 'grey', 'green']

    for data_name, _ in data.items():
        plt.plot(data[data_name], label=data_name)
        # plt.legend()

    plt.legend()
    plt.xlabel('epsilon')
    plt.ylabel(data_name)
    plt.title("Nominal training")
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.001,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=2,
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
