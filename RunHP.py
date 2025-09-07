from datetime import datetime
import os
from math import ceil

import numpy as np
import pandas as pd
import torch
import util
from eval.evalRGGE import valid, precision_at_k, average_precision
from model.RGGE import RGGE
from tqdm import tqdm


def train(model, optimizer, loss_weight_optimizer, DG_train_pos, DG_train_neg, DE_train_pos, DE_train_neg, args, epoch):
    model.train()
    total_loss = 0.0
    device = model.A.device
    type_matrix = util.get_adj_matrix().to(device)
    pred0, label0, pred1, label1 = [], [], [], []
    disLossDG, disLossDE, npLossDG, npLossDE = 0.0, 0.0, 0.0, 0.0
    loaders = [
        util.compute_batches(data, batch_size=ceil(len(data) * args.train_batch_ratio), shuffle=True)
        for data in [DG_train_pos, DG_train_neg, DE_train_pos, DE_train_neg]
    ]

    for DG_pos, DG_neg, DE_pos, DE_neg in zip(*loaders):
        optimizer.zero_grad()
        train_edges = torch.vstack([DG_pos, DG_neg, DE_pos, DE_neg]).to(device)
        DG_true = torch.cat(
            [torch.ones(DG_pos.shape[0], dtype=torch.int), torch.zeros(DG_neg.shape[0], dtype=torch.int)]).to(device)
        DE_true = torch.cat(
            [torch.ones(DE_pos.shape[0], dtype=torch.int), torch.zeros(DE_neg.shape[0], dtype=torch.int)]).to(device)
        entity, R = model(DG_pos, DE_pos)
        disScore = model.distmult(entity, train_edges, type_matrix[train_edges[:, 0], train_edges[:, 1]].to(device))
        # Compute loss
        pos_weight_dg = torch.tensor([len(DG_neg) / len(DG_pos)], device=device)
        pos_weight_de = torch.tensor([len(DE_neg) / len(DE_pos)], device=device)
        criterion_dg = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_dg)
        criterion_de = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_de)
        disLossDG = criterion_dg(disScore[:len(DG_true)], DG_true.float())
        disLossDE = criterion_de(disScore[len(DG_true):], DE_true.float())
        npLossDG = util.n_pair_loss(R[tuple(DG_pos.t())], R[tuple(DG_neg.t())])
        npLossDE = util.n_pair_loss(R[tuple(DE_pos.t())], R[tuple(DE_neg.t())])
        w1, w2, w3, w4 = args.loss_weights
        weighted_loss = torch.exp(-w1) * disLossDG + torch.exp(-w2) * disLossDE + torch.exp(-w3) * npLossDG + torch.exp(-w4) * npLossDE + w1 + w2 + w3 + w4
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clip
        total_loss += weighted_loss
        nan_grad = False
        for name, param in model.named_parameters():
            if torch.isnan(param.grad).any():
                nan_grad = True
                break
        if not nan_grad:
            optimizer.step()
            loss_weight_optimizer.step()
        pred0.append(R[tuple(torch.vstack([DG_pos, DG_neg]).t())])
        label0.append(DG_true)
        pred1.append(R[tuple(torch.vstack([DE_pos, DE_neg]).t())])
        label1.append(DE_true)

    ap0 = average_precision(torch.cat(label0, dim=0).detach().cpu(), torch.cat(pred0, dim=0).detach().cpu())
    ap1 = average_precision(torch.cat(label1, dim=0).detach().cpu(), torch.cat(pred1, dim=0).detach().cpu())
    pk0 = precision_at_k(torch.cat(label0, dim=0).detach().cpu(), torch.cat(pred0, dim=0).detach().cpu(), torch.cat(label0, dim=0).sum().item())
    pk1 = precision_at_k(torch.cat(label1, dim=0).detach().cpu(), torch.cat(pred1, dim=0).detach().cpu(), torch.cat(label1, dim=0).sum().item())

    if epoch == 1:
        final_weights = util.calculate_weights([disLossDG.item(), disLossDE.item(), npLossDG.item(), npLossDE.item()])
        print(f'final_weights: {final_weights}')
        args.loss_weights = torch.nn.Parameter(torch.tensor(final_weights, requires_grad=True))
        w1, w2, w3, w4 = args.loss_weights
        loss = torch.exp(-w1) * disLossDG + torch.exp(-w2) * disLossDE + torch.exp(-w3) * npLossDG + torch.exp(-w4) * npLossDE + w1 + w2 + w3 + w4
        return loss.item(), (ap0+ap1)/2, (pk0+pk1)/2

    return (total_loss*args.train_batch_ratio).item(), (ap0+ap1)/2, (pk0+pk1)/2


def main(DG, DE, args):
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is using ...')
    node_nums = util.get_node_nums()
    num_nodes = sum(node_nums)
    ancillaryData = pd.read_csv(os.path.join(os.getcwd(), f"predata/anc70.csv"))
    ancillaryData = ancillaryData[ancillaryData['idx1'] < ancillaryData['idx2']]
    ancillary = torch.from_numpy(ancillaryData[['idx1', 'idx2']].values)
    keyData = pd.read_csv(os.path.join(os.getcwd(), f"predata/key.csv"))
    key = torch.from_numpy(keyData[['idx1', 'idx2']].values)
    edge_index = torch.cat((key, key[:, [1, 0]]), dim=0)
    edge_weight = torch.ones([edge_index.shape[0], 1], dtype=torch.int)
    A = torch.sparse_coo_tensor(edge_index.t(), edge_weight.squeeze(),
                                size=(num_nodes, num_nodes)).to_dense()
    A.diagonal().copy_((A.sum(dim=1) == 0).float())
    edge_index = torch.cat((ancillary, ancillary[:, [1, 0]], key, key[:, [1, 0]]), dim=0)
    edge_weight = torch.ones([edge_index.shape[0], 1], dtype=torch.int)
    B = torch.sparse_coo_tensor(edge_index.t(), edge_weight.squeeze(),
                                size=(num_nodes, num_nodes)).to_dense()
    B.diagonal().copy_((B.sum(dim=1) == 0).float())
    smiles_sequences_array = np.load('./predata/smiles_sequences.npy')
    pro_sequences_array = np.load('./predata/pro_sequences.npy')
    disease_feature = torch.load('./predata/disease_feature.pt', weights_only=False)
    disease_feature = torch.from_numpy(disease_feature).to(torch.float32)

    hyperparameters = {
        'frtmtl': {
            'A': A,
            'B': B,
            'alpha': args.alpha,
            'beta': args.beta,
            'eta': args.eta,
            'anc': ancillary,
            'disease_feat': disease_feature,
            'add_self_loop': args.add_self_loop,
            'drug_structure': smiles_sequences_array,
            'target_seq': pro_sequences_array,
            'feature_initial_type': args.feature_initial_type,
            'feature_inital_params0': {
                'dict_len': 65 - 1,  # len(CHARCANSMISET)=65
                'embedding_dim': 256,
                'num_filters': 64,
                'filter_length': 4,
                'dropout_rate': 0.5,
                'output_dim': args.in_channels,
            },
            'feature_inital_params1': {
                'dict_len': 25,  # len(CHARPROTSET)=25
                'embedding_dim': 256,
                'num_filters': 64,
                'filter_length': 8,
                'dropout_rate': 0.5,
            },
            'feature_learning_type': args.feature_learning_type,
            'feature_learning_params': {
                'n_relations': args.n_relations,
                'n_bases': args.n_bases,
                'in_channels': args.in_channels,
                'hidden_channels': args.hidden_channels,
                'out_channels': args.out_channels,
                'dropout': args.dropout,
                'num_layers': args.rnum_layers,
            },
            'distmult_loss_type': args.distmult_loss_type,
            'distmult_params': {
                'n_relations': args.n_relations,
                'out_channels': args.out_channels,
            },
            'graph_learning_type': args.graph_learning_type,
            'graph_learning_params': {
                'in_channels': args.out_channels * 2,
                'hidden_channels': args.out_channels,
                'out_channels': 2,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'relu_first': True,
                'batch_norm': True,
                'permutation_invariant': True
            },
            'topological_heuristic_type': args.topological_heuristic_type,
            'topological_heuristic_params': {
                'scaling_parameter': args.scaling_parameter
            },
        },
        'lr': args.lr,
        'epochs': args.epochs,
        'train_batch_ratio': args.train_batch_ratio,
    }
    model = RGGE(**hyperparameters['frtmtl']).to(device)
    args.loss_weights = torch.nn.Parameter(torch.tensor([1., 1., 1., 1.], requires_grad=True))
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': hyperparameters['lr']},
    ], weight_decay=1e-5)
    loss_weight_optimizer = torch.optim.SGD([
        {'params': args.loss_weights, 'lr': args.wlr}
    ])

    del A, disease_feature, edge_index, edge_weight, num_nodes
    del smiles_sequences_array, pro_sequences_array, key, keyData

    best_valid_aupr = -1
    best_epoch = -1
    scores = []
    epoch_iterator = tqdm(range(1, 1 + hyperparameters['epochs']), desc='Epoch')
    for epoch in epoch_iterator:
        train_loss, train_aupr, tpk = train(model, optimizer, loss_weight_optimizer, DG['train']['edge'],
                                                     DG['train']['edge_neg'], DE['train']['edge'], DE['train']['edge_neg'], args, epoch)
        valid_loss, valid_aupr, vpk = valid(model, DG['valid']['edge'], DG['valid']['edge_neg'],
                                                     DE['valid']['edge'], DE['valid']['edge_neg'], args)
        epoch_iterator.set_description(f"Training loss: {train_loss:.4e}, Valid loss: {valid_loss:.4e}, "
                                       f"Valid AUPR: {valid_aupr:.4f}")
        with open(args.log_file, 'a') as f:
            print(f"Epoch = {epoch}:", file=f)
            print(f"Train loss = {train_loss:.4e}", file=f)
            print(f"Train aupr: {train_aupr:.4f}", file=f)
            print(f"Train prec: {tpk:.4f}", file=f)
            print(f"Valid loss = {valid_loss:.4e}", file=f)
            print(f"Valid aupr: {valid_aupr:.4f}", file=f)
            print(f"Valid prec: {vpk:.4f}", file=f)

        scores.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_aupr': train_aupr,
            'train_prec': tpk,
            'valid_loss': valid_loss,
            'valid_aupr': valid_aupr,
            'valid_prec': vpk,
        })

        if valid_aupr > best_valid_aupr:
            best_epoch = epoch
            best_valid_aupr = valid_aupr

    with open(args.log_file, 'a') as f:
        print(f"Best epoch = {best_epoch}", file=f)
        print(f"Loss = {scores[best_epoch - 1]['train_loss']:.4e}", file=f)
        print(f"Valid aupr: {scores[best_epoch - 1]['valid_aupr']:.2%}", file=f)

    return best_valid_aupr


def train_and_evaluate(args):
    # set modifiable hyperparameters in order to adjust the parameters of the model
    split_DG, split_DE = util.split_dataset(args.k_fold, args.random_seed)
    DG = {'train': {}, 'valid': {}, 'test': {}}
    DE = {'train': {}, 'valid': {}, 'test': {}}
    total_aupr = 0.0
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.result_fold = os.path.join(os.getcwd(),
                                    f"RDTD_{current_time}/{args.biased}_{args.loss}_{args.epochs}_{args.alpha}_{args.beta}_{args.eta}/")
    if not os.path.exists(args.result_fold):
        os.makedirs(args.result_fold)
    for (pos_train0, neg_train0, pos_val0, neg_val0, pos_test0, neg_test0,
         pos_train1, neg_train1, pos_val1, neg_val1, pos_test1, neg_test1) in zip(
            split_DG['train']['edge'], split_DG['train']['edge_neg'], split_DG['valid']['edge'],
            split_DG['valid']['edge_neg'], split_DG['test']['edge'], split_DG['test']['edge_neg'],
            split_DE['train']['edge'], split_DE['train']['edge_neg'], split_DE['valid']['edge'],
            split_DE['valid']['edge_neg'], split_DE['test']['edge'], split_DE['test']['edge_neg']):
        DG['train']['edge'] = pos_train0
        DG['train']['edge_neg'] = neg_train0
        DG['valid']['edge'] = pos_val0
        DG['valid']['edge_neg'] = neg_val0
        DG['test']['edge'] = pos_test0
        DG['test']['edge_neg'] = neg_test0
        DE['train']['edge'] = pos_train1
        DE['train']['edge_neg'] = neg_train1
        DE['valid']['edge'] = pos_val1
        DE['valid']['edge_neg'] = neg_val1
        DE['test']['edge'] = pos_test1
        DE['test']['edge_neg'] = neg_test1
        util.set_random_seed(args.random_seed)
        args.log_file = args.result_fold + f'log{args.fold}.txt'
        print(args.log_file)
        del pos_train0, neg_train0, pos_val0, neg_val0, pos_test0, neg_test0
        del pos_train1, neg_train1, pos_val1, neg_val1, pos_test1, neg_test1
        aupr = main(DG, DE, args)
        total_aupr += aupr
        args.fold += 1
    return total_aupr / args.fold
