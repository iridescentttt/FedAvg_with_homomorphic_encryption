import numpy as np
from numpy.core.fromnumeric import size
import torch
import copy
from queue import Queue
import threading


@torch.no_grad()
def mnist_noniid(dataset, num_users):
    """以non-iid的方式划分数据集"""
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


@torch.no_grad()
def FedAvg(w, private_key, device):
    """平均网络权重"""
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = np.add(w_avg[k], w[i][k])
        # w_avg[k] = w_avg[k].tolist()
        # array=np.array(w_avg[k])
        if len(w_avg[k].shape) == 2:
            w_avg[k] = [[private_key.decrypt(x) for x in row]
                        for row in w_avg[k]]
        elif len(w_avg[k].shape) == 1:
            w_avg[k] = [private_key.decrypt(x) for x in w_avg[k]]
        w_avg[k] = torch.Tensor(w_avg[k]).to(device)
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def fedavgJob(client, client_train_epoch, qloss):
    """各个client单独进行训练"""
    loss = []
    for i in range(client_train_epoch):
        loss.append(client.supervisedTrain())
        qloss.put(loss)


def fedavgWork(clients, client_num, client_train_epoch):
    """控制client进行多线程训练"""
    loss_list = []
    qloss = Queue()
    threads = []
    for cid in range(client_num):
        t = threading.Thread(target=fedavgJob,
                             args=(clients[cid], client_train_epoch, qloss))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    for _ in range(client_num):
        loss_list.append(qloss.get())
    return loss_list
