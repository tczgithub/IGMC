import torch
import os
import numpy as np
from model.train import Trainer
from utils import config
from utils import process
import warnings
import random

def main(args):

    # load data
    G_list, X, Y, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    y_true = np.array([np.argmax(l) for l in Y])

    G = G_list[0]
    G2 = G_list[1]
    A, S, R = process.prepare_graph_data(G)
    A2, S2, R2 = process.prepare_graph_data(G2)
    X2 = X
    A = torch.sparse_coo_tensor(indices=torch.tensor([A[0][:, 0], A[0][:, 1]]), values=A[1], size=[args.n_sample, args.n_sample])
    A = A.to_dense()
    X = torch.tensor(X)
    A2 = torch.sparse_coo_tensor(indices=torch.tensor([A2[0][:, 0], A2[0][:, 1]]), values=A2[1], size=[args.n_sample, args.n_sample])
    A2 = A2.to_dense()
    X2 = torch.tensor(X2)

    A = A.to(args.device)
    X = X.to(args.device)
    A2 = A2.to(args.device)
    X2 = X2.to(args.device)

    args.n_sample = X.shape[0]

    feature_dim1 = X.shape[1]
    args.hidden_dims1 = [feature_dim1] + args.hidden_dims1
    feature_dim2 = X2.shape[1]
    args.hidden_dims2 = [feature_dim2] + args.hidden_dims2

    # initial the cluster centers
    train = Trainer(args)
    train.initialization_assign_cluster(A, X, A2, X2)

    # training the model
    train(A, X, S, R, A2, X2, S2, R2, y_true)


def set_rand_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)
    warnings.filterwarnings('ignore')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    args = config.parse_args('imdb')
    # parse param

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    set_rand_seed(args.seed)
    main(args)

