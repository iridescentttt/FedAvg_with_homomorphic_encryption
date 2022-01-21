import sys
import torch
import pyhocon
import random

from dataCenter import *
from utils import *
from models import *
from options import args_parser
from Client import *
from phe import paillier


args = args_parser()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        device_id = torch.cuda.current_device()
        print("using device", device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
# device = "cuda:1"
print("DEVICE:", device, flush=True)

if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=sys.maxsize)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    mode = args.mode
    client_num = args.client_num
    h_feats = args.h_feats
    batch_size = args.batch_size
    lr = args.lr

    # load data
    dataCenter = DataCenter(client_num, device)
    dataCenter.load_dataSet(batch_size)
    print("Load Data Finished ", flush=True)

    train_loader = getattr(dataCenter, "train_loader")
    test_loader = getattr(dataCenter, "test_loader")
    in_feats = getattr(dataCenter, "in_feats")
    num_classes = getattr(dataCenter, "num_classes")

    # init clients
    clients = [
        Client(train_loader[i], mode, in_feats, h_feats, num_classes, lr, device).to(device) for i in range(client_num)
    ]

    print("clients init finished ", flush=True)

    test_f1 = [0 for i in range(client_num)]
    test_avg_f1 = []
    max_test_f1 = [0 for i in range(client_num)]
    max_test_avg_f1 = 0
    client_train_epoch = 5

    """server生成公私钥"""
    public_key, private_key = paillier.generate_paillier_keypair(n_length=100)
    print("Public key: g", public_key.g, "n", public_key.n)
    print("Private key: p", private_key.p, "q", private_key.q)

    """开始训练"""
    for epoch in range(1, args.epochs + 1):
        """fedavg平均权重"""
        if mode == "fedavg":
            with torch.no_grad():
                clients_dict = []
                for i in range(client_num):
                    client_dict = clients[i].state_dict()
                    
                    """对网络权重进行加密"""
                    for key in client_dict.keys():
                        if len(client_dict[key].shape) == 2:
                            client_dict[key] = [[public_key.encrypt(x) for x in row]
                                                for row in client_dict[key].cpu().tolist()]
                        elif len(client_dict[key].shape) == 1:
                            client_dict[key] = [
                                public_key.encrypt(x) for x in client_dict[key].cpu().tolist()]
                    clients_dict += [client_dict]
                """FedAvg操作"""
                w_avg = FedAvg(clients_dict, private_key, device)
                for cid in range(client_num):
                    clients[cid].load_state_dict(w_avg)
                    clients[cid].init_model = w_avg

        """开始训练"""
        loss_list = fedavgWork(clients, client_num, client_train_epoch)

        """test"""
        with torch.no_grad():
            for cid in range(client_num):
                test_f1[cid] = clients[cid].test(test_loader)
            test_avg_f1.append(np.mean(test_f1))
            max_test_avg_f1 = max(max_test_avg_f1, test_avg_f1[-1])

        """打印结果"""
        print("-----epoch", epoch, "test f1:",
              test_avg_f1[-1], " -----", flush=True)

    print("max global f1:", max_test_avg_f1)
