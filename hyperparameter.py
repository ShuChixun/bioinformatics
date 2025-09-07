import argparse
from ast import literal_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--result_fold', default='result',
                        help='Result fold. ')
    parser.add_argument('--alpha', type=float, default=0.,
                        help='Topological weight. Default is 0.0. ')
    parser.add_argument('--beta', type=float, default=1.,
                        help='Trained weight. Default is 1.0. ')
    parser.add_argument('--eta', type=float, default=0.,
                        help='Proportion of added edges. Default is 0.0. ')
    parser.add_argument('--add-self-loop', type=literal_eval, default=False,
                        help='Whether to add self-loops to all nodes. Default is False. ')
    parser.add_argument('--graph-learning-type', default='mlp',
                        help='Type of the graph learning component. Default is mlp. ')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of layers in mlp. Default is 3. ')
    parser.add_argument('--rnum-layers', type=int, default=2,
                        help='Number of layers in rgcn. Default is 2. ')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate. Default is 0.5. ')
    parser.add_argument('--topological-heuristic-type', default='ac',
                        help='Type of the topological heuristic component. Default is ac. ')
    parser.add_argument('--scaling-parameter', type=int, default=3,
                        help='Scaling parameter of ac. Default is 3. ')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of epochs. Default is 100. ')
    parser.add_argument('--wlr', type=float, default=1e-4,
                        help='Learning rate. Default is 1e-4.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate. Default is 1e-3. ')
    parser.add_argument('--train-batch-ratio', type=float, default=0.1,
                        help='Ratio of training edges per train batch. Default is 0.1. ')
    parser.add_argument('--patience', type=int, default=250,
                        help='Patience for early stopping. Default is 25. ')
    parser.add_argument('--feature_initial_type', type=str, default='cnn',
                        help='Type of the feature initial component. Default is cnn. ')
    parser.add_argument('--distmult_loss_type', type=str, default='Distmult',
                        help='Type of the distmult loss. Default is distmult. ')
    parser.add_argument('--feature_learning_type', type=str, default='rgcn', help='feature_learning_type')
    parser.add_argument('--n_relations', type=int, default=6, help='n_relations')
    parser.add_argument('--n_bases', type=int, default=30, help='n_bases')
    parser.add_argument('--in_channels', type=int, default=128, help='in_channels')
    parser.add_argument('--hidden_channels', type=int, default=128, help='hidden_channels') #
    parser.add_argument('--out_channels', type=int, default=64, help='out_channels')
    parser.add_argument('--random_seed', type=int, default=523,
                        help='Random seed for training. Default is 523. ')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Index of cuda device to use. Default is 0. ')
    parser.add_argument('--k_fold', type=int, default=10,
                        help='Number of folds for k-fold cross validation. Default is 10. ')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold for training. Default is 0. ')
    parser.add_argument('--biased', type=str, default='unbiased',
                        help='Whether to use biased sampling. Default is False. ')
    parser.add_argument('--loss', type=str, default='NP',
                        help='Loss function. Default is NP. ')

    return parser.parse_args()
