import numpy as np
import scipy.io
import torch

def load_data(data_str):
    bus_no = 118
    bus_data = scipy.io.loadmat(data_str)
    data_inputs = bus_data['input_nodal'].astype(np.float32)
    data_labels = bus_data['labels'].astype(np.float32)
    data_adj = bus_data['adj']

    window_size = np.minimum(10, data_inputs.shape[1])
    train_portion = int(0.8 * window_size)
    x_train = data_inputs[:, :train_portion]
    y_train = data_labels[:, :train_portion]
    x_test = data_inputs[:, train_portion:]
    y_test = data_labels[:, train_portion:]

    train_mask = np.array(np.ones(bus_no), dtype=np.bool)
    test_mask = np.array(np.ones(bus_no), dtype=np.bool)

    features_v = data_inputs[:bus_no]
    features_p = data_inputs[bus_no:2*bus_no]
    features_q = data_inputs[2*bus_no: ]
    labels_mag = data_labels[0:bus_no]
    labels_ang = data_labels[bus_no:]
    features_stack = np.stack([features_v[:, iter], features_p[:, iter], features_q[:, iter]], axis=1)
    labels_ang_rad = np.dot(np.pi / 180.0, labels_ang[:, iter])
    labels_stack = np.stack([labels_mag[:, iter] * np.cos(labels_ang_rad), labels_mag[:, iter] * np.sin(labels_ang_rad)], axis=1).astype(np.float32)
    return torch.from_numpy(features_stack), torch.from_numpy(labels_stack), data_adj, y_train, y_test, train_mask, test_mask, train_portion