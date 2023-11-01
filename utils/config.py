import argparse


def parse_args(dataset='imdb'):
    """
    Parses the arguments.
    """
    parser = argparse.ArgumentParser(description="---")

    if dataset == 'imdb':
        parser.add_argument('--dataset', nargs='?', default='imdb', help='Input dataset')
        parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default is 1e-3')
        parser.add_argument('--seed', type=int, default=9, help='Seed for fixing the results')
        parser.add_argument('--n_epochs', default=300, type=int, help='Number of epochs')
        parser.add_argument('--hidden_dims1', type=list, nargs='+', default=[512, 256], help='Number of dimensions1')
        parser.add_argument('--hidden_dims2', type=list, nargs='+', default=[512, 256], help='Number of dimensions2')
        parser.add_argument('--embedding', type=int, default=256)
        parser.add_argument('--lambda_1', default=1, type=float)
        parser.add_argument('--lambda_2', default=1, type=float)
        parser.add_argument('--lambda_3', default=0.1, type=float)
        parser.add_argument('--beta', default=1, type=float, help='-----')
        parser.add_argument('--ADJ', default='A_A2', type=str, help='-----')
        parser.add_argument('--cluster', default=3, type=float, help='The number of clusters')
        parser.add_argument('--n_sample', type=int, default=4780, help='The number of the samples')
        parser.add_argument('--tau', default=1.0, type=float, help='Dropout')
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--init', default=72, type=int, help='Fix initial centroids')
        parser.add_argument('--weight_decay', default=1e-5, type=float, help='Gradient clipping')

    if dataset == 'acm':
        parser.add_argument('--dataset', nargs='?', default='acm', help='Input dataset')
        parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default is 1e-3')
        parser.add_argument('--seed', type=int, default=2, help='Seed for fixing the results')
        parser.add_argument('--n_epochs', default=50, type=int, help='Number of epochs')
        parser.add_argument('--hidden_dims1', type=list, nargs='+', default=[786, 256], help='Number of dimensions1')
        parser.add_argument('--hidden_dims2', type=list, nargs='+', default=[786, 256], help='Number of dimensions2')
        parser.add_argument('--embedding', type=int, default=256)
        parser.add_argument('--lambda_1', default=1, type=float)
        parser.add_argument('--lambda_2', default=1, type=float)
        parser.add_argument('--lambda_3', default=0.1, type=float)
        parser.add_argument('--cluster', default=3, type=float, help='The number of clusters')
        parser.add_argument('--beta', default=1, type=float, help='-----')
        parser.add_argument('--ADJ', default='A_A2', type=str, help='-----')
        parser.add_argument('--n_sample', type=int, default=3025, help='The number of the samples')
        parser.add_argument('--tau', default=1.0, type=float, help='Dropout')
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--init', default=72, type=int, help='Fix initial centroids')
        parser.add_argument('--weight_decay', default=1e-5, type=float, help='Gradient clipping')


    return parser.parse_args()

