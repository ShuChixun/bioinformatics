import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

import torch


def seq_trans(line, max_len, dict_idx):
    x = np.zeros(max_len)
    for i, ch in enumerate(line[:max_len]):
        x[i] = dict_idx[ch]
    return x


def get_adj_matrix():
    # After saving the calculation, directly read the pt file.
    if os.path.exists(os.path.join(os.getcwd(), 'predata/adj_matrix.pt')):
        adj_matrix = torch.load(os.path.join(os.getcwd(), 'predata/adj_matrix.pt'), weights_only= True)
    else:
        lenDrug, lenTarget, lenDisease = get_node_nums()
        total_nodes = lenDrug + lenTarget + lenDisease
        adj_matrix = torch.zeros((total_nodes, total_nodes), dtype=torch.long)
        # Drug-Drug type is 0.
        adj_matrix[:lenDrug, :lenDrug] = 0
        # Drug-Gene type is 1
        adj_matrix[:lenDrug, lenDrug:lenDrug + lenTarget] = 1
        adj_matrix[lenDrug:lenDrug + lenTarget, :lenDrug] = 1
        # Drug-Disease type is 2
        adj_matrix[:lenDrug, lenDrug + lenTarget:] = 2
        adj_matrix[lenDrug + lenTarget:, :lenDrug] = 2
        # Gene-Gene type is 3
        adj_matrix[lenDrug:lenDrug + lenTarget, lenDrug:lenDrug + lenTarget] = 3
        # Gene-Disease type is 4
        adj_matrix[lenDrug:lenDrug + lenTarget, lenDrug + lenTarget:] = 4
        adj_matrix[lenDrug + lenTarget:, lenDrug:lenDrug + lenTarget] = 4
        # Disease-Disease type is 5
        adj_matrix[lenDrug + lenTarget:, lenDrug + lenTarget:] = 5
        # save adj_matrix
        torch.save(adj_matrix, os.path.join(os.getcwd(), 'predata/adj_matrix.pt'))

    return adj_matrix


def get_node_nums():
    lenDrug = 708
    lenTarget = 1512
    lenDisease = 5603
    node_nums = [lenDrug, lenTarget, lenDisease]

    return node_nums


def calculate_weights(base_ratio):
    normalization_factor = base_ratio[-1]  # Use the last weight as the normalization factor.
    normalized_weights = [w / normalization_factor for w in base_ratio]
    trans_weights = [np.log(weight)+1 for weight in normalized_weights]
    clamped_weights = [max(0.1, min(2.0, w)) for w in trans_weights]
    final_weights = [round(w, 1) for w in clamped_weights]

    return final_weights


def split_dataset(k_fold=10, random_seed=0):  # unbiased split method
    np.random.seed(random_seed)
    d, t, e = 708, 1512, 5603
    keyData = pd.read_csv(os.path.join(os.getcwd(), f"predata/key.csv"))
    keyData = keyData[keyData['idx1'] < keyData['idx2']]
    print('Original positive edge ratio', len(keyData)/(d*(t+e)))
    n = d + t + e
    DG = keyData[keyData['type'] == 1]
    DG = torch.from_numpy(DG[['idx1', 'idx2']].values)
    DE = keyData[keyData['type'] == 2]
    DE = torch.from_numpy(DE[['idx1', 'idx2']].values)

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
    # Get DG train, valid, test dataset
    indices_DG = np.arange(DG.shape[0])
    np.random.shuffle(indices_DG)
    pos_DG = DG[indices_DG]

    split_DG = {'train': {'edge': [], 'edge_neg': []}, 'valid': {'edge': [], 'edge_neg': []},
                'test': {'edge': [], 'edge_neg': []}}
    # positive DG
    for train_indices_DG, test_indices_DG in kf.split(pos_DG):
        pos_test = pos_DG[test_indices_DG]
        val_DG_size = len(test_indices_DG) // 2
        pos_val = pos_DG[train_indices_DG][:val_DG_size]
        pos_train = pos_DG[train_indices_DG][val_DG_size:]
        split_DG['train']['edge'].append(pos_train)
        split_DG['valid']['edge'].append(pos_val)
        split_DG['test']['edge'].append(pos_test)

        # negative DG
        drug_nums = d
        gene_nums = t
        train_edge_neg_mask = torch.zeros((n, n), dtype=torch.bool)
        train_edge_neg_mask[:drug_nums, drug_nums:drug_nums + gene_nums] = True
        train_edge_neg_mask[tuple(pos_train.T.tolist())] = False
        split_DG['train']['edge_neg'].append(torch.nonzero(train_edge_neg_mask))
        valid_edge_neg_mask = train_edge_neg_mask.clone()
        valid_edge_neg_mask[tuple(pos_val.T.tolist())] = False
        split_DG['valid']['edge_neg'].append(torch.nonzero(valid_edge_neg_mask))
        test_edge_neg_mask = valid_edge_neg_mask.clone()
        test_edge_neg_mask[tuple(pos_test.T.tolist())] = False
        split_DG['test']['edge_neg'].append(torch.nonzero(test_edge_neg_mask))

    indices_DE = np.arange(DE.shape[0])
    np.random.shuffle(indices_DE)
    pos_DE = DE[indices_DE]

    split_DE = {'train': {'edge': [], 'edge_neg': []}, 'valid': {'edge': [], 'edge_neg': []},
                'test': {'edge': [], 'edge_neg': []}}
    # positive DE
    for train_indices_DE, test_indices_DE in kf.split(pos_DE):
        pos_test = pos_DE[test_indices_DE]
        val_DE_size = len(test_indices_DE) // 2
        pos_val = pos_DE[train_indices_DE][:val_DE_size]
        pos_train = pos_DE[train_indices_DE][val_DE_size:]
        split_DE['train']['edge'].append(pos_train)
        split_DE['valid']['edge'].append(pos_val)
        split_DE['test']['edge'].append(pos_test)

        # negative DG
        drug_nums = d
        target_nums = t
        train_edge_neg_mask = torch.zeros((n, n), dtype=torch.bool)
        train_edge_neg_mask[:drug_nums, drug_nums + target_nums:] = True
        train_edge_neg_mask[tuple(pos_train.T.tolist())] = False
        split_DE['train']['edge_neg'].append(torch.nonzero(train_edge_neg_mask))
        valid_edge_neg_mask = train_edge_neg_mask.clone()
        valid_edge_neg_mask[tuple(pos_val.T.tolist())] = False
        split_DE['valid']['edge_neg'].append(torch.nonzero(valid_edge_neg_mask))
        test_edge_neg_mask = valid_edge_neg_mask.clone()
        test_edge_neg_mask[tuple(pos_test.T.tolist())] = False
        split_DE['test']['edge_neg'].append(torch.nonzero(test_edge_neg_mask))

    return split_DG, split_DE


def n_pair_loss(out_pos, out_neg):
    agg_size = out_neg.shape[0] // out_pos.shape[0]
    agg_size_p1 = agg_size + 1
    agg_size_p1_count = out_neg.shape[0] % out_pos.shape[0]
    out_pos_agg_p1 = out_pos[:agg_size_p1_count].unsqueeze(-1)
    out_pos_agg = out_pos[agg_size_p1_count:].unsqueeze(-1)
    out_neg_agg_p1 = out_neg[:agg_size_p1_count * agg_size_p1].reshape(-1, agg_size_p1)
    out_neg_agg = out_neg[agg_size_p1_count * agg_size_p1:].reshape(-1, agg_size)
    out_diff_agg_p1 = out_neg_agg_p1 - out_pos_agg_p1
    out_diff_agg = out_neg_agg - out_pos_agg
    out_diff_exp_sum_p1 = torch.exp(torch.clamp(out_diff_agg_p1, max=80.0)).sum(axis=1)
    out_diff_exp_sum = torch.exp(torch.clamp(out_diff_agg, max=80.0)).sum(axis=1)
    out_diff_exp_cat = torch.cat([out_diff_exp_sum_p1, out_diff_exp_sum])
    loss = torch.log(1 + out_diff_exp_cat).sum() / (len(out_pos) + len(out_neg))

    return loss


def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_batches(rows, batch_size, shuffle=True):
    if shuffle:
        return torch.split(rows[torch.randperm(rows.shape[0])], batch_size)
    else:
        return torch.split(rows, batch_size)


def plot_train_curve(scores, filename, fold):
    epochs = [score['epoch'] for score in scores]
    train_loss = [score['train_loss'] for score in scores]
    train_prec = [score['train_prec']*100 for score in scores]
    valid_loss = [score['valid_loss'] for score in scores]
    valid_prec = [score['valid_prec']*100 for score in scores]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot train loss on the left y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='red')
    ax1.plot(epochs, train_loss, label='Train Loss', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot train precision on the right y-axis
    ax2.set_ylabel('Train AUPR', color='blue')
    # ax2.plot(epochs, train_aupr, label='Train AUPR', color='blue')
    ax2.plot(epochs, train_prec, label='Train prec', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Set title and save the figure
    # plt.title('Train Loss and Train AUPR over Epochs')
    fig.tight_layout()  # Adjust layout to prevent clipping of ylabel
    plt.savefig(filename + f'/train_loss_train_aupr_{fold}.png')
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot validation loss on the left y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Valid Loss', color='red')
    ax1.plot(epochs, valid_loss, label='Valid Loss', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot validation precision on the right y-axis
    ax2.set_ylabel('Valid AUPR', color='blue')
    # ax2.plot(epochs, valid_aupr, label='Valid AUPR', color='blue')
    ax2.plot(epochs, valid_prec, label='Valid prec', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Set title and save the figure
    # plt.title('Valid Loss and Validation AUPR over Epochs')
    fig.tight_layout()  # Adjust layout to prevent clipping of ylabel
    plt.savefig(filename + f'/valid_loss_valid_aupr_{fold}.png')
    plt.close(fig)


def plot_pks_hitsks(pks, hitsks, results_folder, fold):
    pks = [pk * 100 for pk in pks]
    hitsks = [hits * 100 for hits in hitsks]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    plt.figure(figsize=(12, 6))
    x_labels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.plot(x_labels, pks, marker='o', color='red', label='Test precision at k')
    plt.title('Test Precision at Different Percentages')
    plt.xlabel('Percentage')
    plt.ylabel('Test Precision (%)')
    plt.legend()
    plt.grid(True)
    pks_plot_path = os.path.join(results_folder, f'fold_{fold}_test_pks.png')
    plt.savefig(pks_plot_path)
    plt.close()

    # 绘制 hitsks 折线图
    plt.figure(figsize=(12, 6))
    x_labels_hitsks = [r'$2^{-2}$', r'$2^{-1}$', r'$2^{0}$', r'$2^{1}$', r'$2^{2}$', r'$2^{3}$', r'$2^{4}$',
                       r'$2^{5}$']
    plt.plot(x_labels_hitsks, hitsks, marker='v', color='green', label='Test hits at k')
    plt.title('Test Hits at Different Counts')
    plt.xlabel('Count')
    plt.ylabel('Test Hits (%)')
    plt.legend()
    plt.grid(True)
    hitsks_plot_path = os.path.join(results_folder, f'fold_{fold}_test_hitsks.png')
    plt.savefig(hitsks_plot_path)
    plt.close()
