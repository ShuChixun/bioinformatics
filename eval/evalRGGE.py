import os

import pandas as pd
import torch
from math import ceil
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, \
    accuracy_score, matthews_corrcoef

import util


def precision_at_k(eval_true, eval_pred, k):
    """
    Compute precision@k.

    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :param k: k value.
    :return: Precision@k
    """

    eval_top_index = torch.topk(eval_pred, k, sorted=False).indices.cpu()
    eval_tp = eval_true[eval_top_index].sum().item()
    pk = eval_tp / k

    return pk


def hits_at_k(eval_true, eval_pred, k):
    """
    Compute hits@k.

    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :param k: k value.
    :return: Hits@k.
    """

    pred_pos = eval_pred[eval_true == 1]
    pred_neg = eval_pred[eval_true == 0]
    kth_score_in_negative_edges = torch.topk(pred_neg, k)[0][-1]
    hitsk = float(torch.sum(pred_pos > kth_score_in_negative_edges).cpu()) / len(pred_pos)

    return hitsk


def average_precision(eval_true, eval_pred):
    """
    Compute Average Precision (AP).

    :param eval_true: Evaluation labels.
    :param eval_pred: Evaluation predictions.
    :return: AP.
    """

    return average_precision_score(eval_true, eval_pred)


@torch.no_grad()
def valid(model, DG_valid_pos, DG_valid_neg, DE_valid_pos, DE_valid_neg, args):
    model.eval()
    raw_loss = 0.0
    losses = {'loss0': [], 'loss1': [], 'loss2': [], 'loss3': []}
    type_matrix = util.get_adj_matrix().to(model.A.device)
    valid_edges = torch.cat([DG_valid_pos, DG_valid_neg, DE_valid_pos, DE_valid_neg]).to(model.A.device)
    entity, R = model(DG_valid_pos, DE_valid_pos)
    disScore = model.distmult(entity, valid_edges, type_matrix[valid_edges[:, 0], valid_edges[:, 1]].to(model.A.device))
    # criterion = torch.nn.BCEWithLogitsLoss()
    DG_true = torch.cat([torch.ones(DG_valid_pos.shape[0], dtype=torch.int), torch.zeros(DG_valid_neg.shape[0], dtype=torch.int)]).to(model.A.device)
    # disLossDG = criterion(disScore[:len(DG_true)], DG_true.float())  # 计算DG这种正边和负边的损失
    DE_true = torch.cat([torch.ones(DE_valid_pos.shape[0], dtype=torch.int), torch.zeros(DE_valid_neg.shape[0], dtype=torch.int)]).to(model.A.device)
    # disLossDE = criterion(disScore[len(DG_true):], DE_true.float())  # 计算DE这种正边和负边的损失
    # disLossDG = util.n_pair_loss(disScore[:len(DG_valid_pos)], disScore[len(DG_valid_pos):len(DG_true)])
    pos_weight_dg = torch.tensor([len(DG_valid_neg) / len(DG_valid_pos)], device=model.A.device)
    pos_weight_de = torch.tensor([len(DE_valid_neg) / len(DE_valid_pos)], device=model.A.device)
    criterion_dg = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_dg)
    criterion_de = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_de)

    # 假设 disScore 是 DistMult 输出的原始 logits
    disLossDG = criterion_dg(disScore[:len(DG_true)], DG_true.float())
    disLossDE = criterion_de(disScore[len(DG_true):], DE_true.float())
    # disLossDE = util.n_pair_loss(disScore[len(DG_true):len(DG_true) + len(DE_valid_pos)],
    #                              disScore[len(DG_true) + len(DE_valid_pos):])
    npLossDG, npLossDE = None, None
    if args.loss == 'NP':
        npLossDG = util.n_pair_loss(R[tuple(DG_valid_pos.t())], R[tuple(DG_valid_neg.t())])
        npLossDE = util.n_pair_loss(R[tuple(DE_valid_pos.t())], R[tuple(DE_valid_neg.t())])
    elif args.loss == 'CE':
        disScore = R[tuple(valid_edges.t())]
        npLossDG = criterion_dg(disScore[:len(DG_true)], DG_true.float())
        npLossDE = criterion_de(disScore[len(DG_true):], DE_true.float())
    # print(disLossDG, disLossDE, npLossDG, npLossDE)
    w1, w2, w3, w4 = args.loss_weights
    # loss = ((1 / (2 * w1 ** 2)) * disLossDG + (1 / (2 * w2 ** 2)) * disLossDE + (1 / (2*w3 ** 2)) * npLossDG + (1 / (2*w4 ** 2)) * npLossDE + torch.log(w1) + torch.log(w2) + torch.log(w3) + torch.log(w4))
    loss = torch.exp(-w1) * disLossDG + torch.exp(-w2) * disLossDE + torch.exp(-w3) * npLossDG + torch.exp(-w4) * npLossDE + w1 + w2 + w3 + w4
    # loss = disLossDG + disLossDE + npLossDG + npLossDE
    raw_loss += (disLossDG + disLossDE + npLossDG + npLossDE)
    # valid_pred = R[tuple(valid_edges.t())]
    # valid_true = torch.cat([DG_true, DE_true])
    # pk = precision_at_k(valid_true, valid_pred, valid_true.sum().item())
    ap0 = average_precision(DG_true.detach().cpu(), R[tuple(torch.cat([DG_valid_pos, DG_valid_neg]).t())].detach().cpu())
    ap1 = average_precision(DE_true.detach().cpu(), R[tuple(torch.cat([DE_valid_pos, DE_valid_neg]).t())].detach().cpu())
    losses['loss0'].append(disLossDG.item())
    losses['loss1'].append(disLossDE.item())
    losses['loss2'].append(npLossDG.item())
    losses['loss3'].append(npLossDE.item())

    pk0 = precision_at_k(DG_true.detach().cpu(), R[tuple(torch.cat([DG_valid_pos, DG_valid_neg]).t())].detach().cpu(), DG_true.sum().item())
    pk1 = precision_at_k(DE_true.detach().cpu(), R[tuple(torch.cat([DE_valid_pos, DE_valid_neg]).t())].detach().cpu(), DE_true.sum().item())

    return loss.item(), (ap0+ap1)/2, losses, pk0, pk1, raw_loss.item()


@torch.no_grad()
def test(model, DG_test_pos, DG_test_neg, DE_test_pos, DE_test_neg, args, log_file, scores):
    model.load_state_dict(
        torch.load(args.result_fold + f'model_checkpoint_{args.fold}.pth', map_location=args.device, weights_only=True))

    model.eval()
    test_edges = torch.cat([DG_test_pos, DG_test_neg, DE_test_pos, DE_test_neg]).to(model.A.device)
    DG_true = torch.cat([torch.ones(DG_test_pos.shape[0], dtype=torch.int), torch.zeros(DG_test_neg.shape[0], dtype=torch.int)])
    DE_true = torch.cat([torch.ones(DE_test_pos.shape[0], dtype=torch.int), torch.zeros(DE_test_neg.shape[0], dtype=torch.int)])
    _, R = model(DG_test_pos, DE_test_pos)
    test_pred = R[tuple(test_edges.t())]
    pred = test_pred.detach().cpu()
    pred = pred.sigmoid()

    DG_pred = pred[:DG_true.shape[0]]
    DG_pred_binary = (DG_pred > 0.5).int()
    num_DG_edges = DG_true.sum().item()
    # save predicted score and true label of DG test
    # results_df = pd.DataFrame({
    #     'true_label': DG_true,
    #     'predict': DG_pred,
    # })
    # results_df.to_csv(os.path.join(args.result_fold, f'fold{args.fold}DG.csv'), index=False)
    auc = roc_auc_score(DG_true, DG_pred)
    ap = average_precision(DG_true, DG_pred)
    dg_precision = precision_score(DG_true, DG_pred_binary)
    dg_recall = recall_score(DG_true, DG_pred_binary)
    dg_f1 = f1_score(DG_true, DG_pred_binary)
    dg_accuracy = accuracy_score(DG_true, DG_pred_binary)
    dg_mcc = matthews_corrcoef(DG_true, DG_pred_binary)
    pks = [precision_at_k(DG_true, DG_pred, k) for k in
           [ceil(num_DG_edges * ratio) for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]]
    hitsks = [hits_at_k(DG_true, DG_pred, k) for k in (25, 50, 100, 200, 400, 800, 1600, 3200)]
    print('DG', auc, ap, pks, hitsks)
    with open(log_file, 'a') as f:
        print(f"DG AUC: {auc:.2%}, AP: {ap:.2%}", file=f)
        print(f"DG precision: {dg_precision:.2%}, recall: {dg_recall:.2%}, f1: {dg_f1:.2%}, accuracy: {dg_accuracy:.2%}, mcc: {dg_mcc:.2%}", file=f)
        for i, k in enumerate(('10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%')):
            print(f"Test precision@{k}: {pks[i]:.2%}", file=f)
        for i, k in enumerate((25, 50, 100, 200, 400, 800, 1600, 3200)):
            print(f"Test hits@{k}: {hitsks[i]:.2%}", file=f)
        print(file=f)
    # draw pks and hitsks
    # util.plot_pks_hitsks(pks, hitsks, args.result_fold, f'DG{args.fold}')

    DE_pred = pred[DG_true.shape[0]:]
    DE_pred_binary = (DE_pred > 0.5).int()
    num_DE_edges = DE_true.sum().item()
    # save predicted score and true label of DE test
    results_df = pd.DataFrame({
        'true_label': DE_true,
        'predict': DE_pred,
    })
    results_df.to_csv(os.path.join(args.result_fold, f'fold{args.fold}DE.csv'), index=False)
    auc = roc_auc_score(DE_true, DE_pred)
    ap = average_precision(DE_true, DE_pred)
    de_precision = precision_score(DE_true, DE_pred_binary)
    de_recall = recall_score(DE_true, DE_pred_binary)
    de_f1 = f1_score(DE_true, DE_pred_binary)
    de_accuracy = accuracy_score(DE_true, DE_pred_binary)
    de_mcc = matthews_corrcoef(DE_true, DE_pred_binary)
    pks = [precision_at_k(DE_true, DE_pred, k) for k in
           [ceil(num_DE_edges * ratio) for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]]
    hitsks = [hits_at_k(DE_true, DE_pred, k) for k in (25, 50, 100, 200, 400, 800, 1600, 3200)]
    print('DE', auc, ap, pks, hitsks)
    with open(log_file, 'a') as f:
        print(f"DE AUC: {auc:.2%}, AP: {ap:.2%}", file=f)
        print(f"DE precision: {de_precision:.2%}, recall: {de_recall:.2%}, f1: {de_f1:.2%}, accuracy: {de_accuracy:.2%}, mcc: {de_mcc:.2%}", file=f)
        for i, k in enumerate(('10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%')):
            print(f"Test precision@{k}: {pks[i]:.2%}", file=f)
        for i, k in enumerate((25, 50, 100, 200, 400, 800, 1600, 3200)):
            print(f"Test hits@{k}: {hitsks[i]:.2%}", file=f)
        print(file=f)
    # draw pks and hitsks
    # util.plot_pks_hitsks(pks, hitsks, args.result_fold, f'DE{args.fold}')
