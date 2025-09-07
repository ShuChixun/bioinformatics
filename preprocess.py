import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from util import seq_trans, get_node_nums
from node2vec import node2vec_main


def get_node_types(node_nums):
    a, b, c = node_nums
    offsets = {'drug': a}
    offsets['target'] = offsets['drug'] + b
    offsets['disease'] = offsets['target'] + c

    node_types = np.zeros((offsets['disease'],), dtype=np.int32)
    node_types[offsets['drug']:offsets['target']] = 1
    node_types[offsets['target']:offsets['disease']] = 2

    return offsets


def get_all_indices(offsets, dd, dt, de, te, tt, ee):
    dd['type'] = 0
    print('dd.shape', dd.shape)
    print('sparse ratio', len(dd)/(708*708))
    dt['type'] = 1
    dt = dt[dt['weight'] == 1]
    dt.loc[:, 'idx2'] = dt['idx2'] + offsets['drug']
    print('dt.shape', dt.shape)
    print('sparse ratio', len(dt)/(708*1512))
    de['type'] = 2
    de.loc[:, 'idx2'] = de['idx2'] + offsets['target']
    print('de.shape', de.shape)
    print('sparse ratio', len(de)/(708*5603))
    tt['type'] = 3
    tt.loc[:, 'idx1'] = tt['idx1'] + offsets['drug']
    tt.loc[:, 'idx2'] = tt['idx2'] + offsets['drug']
    print('tt.shape', tt.shape)
    print('sparse ratio', len(tt)/(1512*1512))
    te['type'] = 4
    te.loc[:, 'idx1'] = te['idx1'] + offsets['drug']
    te.loc[:, 'idx2'] = te['idx2'] + offsets['target']
    print('te.shape', te.shape)
    print('sparse ratio', len(te)/(1512*5603))
    ee.loc[:, 'idx1'] = ee['idx1'] + offsets['target']
    ee.loc[:, 'idx2'] = ee['idx2'] + offsets['target']
    # 去除对角线
    ee = ee[ee['idx1'] != ee['idx2']]
    print('ee.shape', ee.shape)
    print('sparse ratio', len(ee)/(5603*5603))
    all_indices = pd.concat([dd, dt, de, tt, te, ee], ignore_index=True)
    print('all_df.shape', all_indices.shape)
    print('all sparse ratio', len(all_indices)/((708+1512+5603)**2)*2)
    anc = pd.concat([dd, tt, te, ee], ignore_index=True)
    anc.to_csv('./predata/anc70.csv', index=False)
    key = pd.concat([dt, de], ignore_index=True)
    key.to_csv('./predata/key.csv', index=False)


def getee(de, te, node_nums, threshold):
    d, t, e = node_nums
    deMatrix = csr_matrix((np.ones(len(de)), (de['idx2'], de['idx1'])), shape=(e, d))
    teMatrix = csr_matrix((np.ones(len(te)), (te['idx2'], te['idx1'])), shape=(e, t))
    jaccardSimde = 1 - squareform(pdist(deMatrix.toarray(), metric='jaccard'))
    jaccardSimte = 1 - squareform(pdist(teMatrix.toarray(), metric='jaccard'))
    sim_diseases = np.round((jaccardSimde + jaccardSimte)/2, 4).astype(np.float32)
    np.save('./predata/sim_diseases.npy', sim_diseases)
    ee = (jaccardSimde > threshold) | (jaccardSimte > threshold)
    rows, cols = np.where(ee)
    mask = ee[rows, cols]
    rows = rows[mask]
    cols = cols[mask]
    return pd.DataFrame({'idx1': rows, 'idx2': cols, 'weight': 1, 'type': 5})


def getDiseaseFeature(de, node_nums):
    d, t, e = node_nums
    de_npy = de.to_numpy()[:, :2]
    de_matrix = np.zeros((d, e), dtype=int)
    de_matrix[de_npy[:, 0], de_npy[:, 1]] = 1
    feature = node2vec_main(de_matrix, type='H')
    print(feature[d:].shape)
    torch.save(feature[d:], './predata/disease_feature.pt')


def preprocess():
    start_time = time.time()  # 预计280s
    prefix = os.path.join(os.getcwd(), "Luo")
    dd = pd.read_csv(os.path.join(prefix, "drug_drug.dat"), encoding='utf-8', delimiter=',',
                     names=['idx1', 'idx2', 'weight']).reset_index(drop=True)
    dt = pd.read_csv(os.path.join(prefix, "drug_target.dat"), encoding='utf-8', delimiter=',',
                     names=['idx1', 'idx2', 'weight']).reset_index(drop=True)
    de = pd.read_csv(os.path.join(prefix, "drug_dis.dat"), encoding='utf-8', delimiter=',',
                     names=['idx1', 'idx2', 'weight']).reset_index(drop=True)
    te = pd.read_csv(os.path.join(prefix, "protein_dis.dat"), encoding='utf-8', delimiter=',',
                     names=['idx1', 'idx2', 'weight']).reset_index(drop=True)
    tt = pd.read_csv(os.path.join(prefix, "pro_pro.dat"), encoding='utf-8', delimiter=',',
                     names=['idx1', 'idx2', 'weight']).reset_index(drop=True)
    folder_name = os.path.join(os.getcwd(), "predata/")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    threshold = 0.7
    node_nums = get_node_nums()
    # getDiseaseFeature(de, node_nums)  # It spends a lot of time, please comment if calculations are not necessary.
    ee = getee(de, te, node_nums, threshold)
    offsets = get_node_types(node_nums)
    get_all_indices(offsets, dd, dt, de, te, tt, ee)
    CHARCANSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                     "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, "A": 19, "C": 20, "B": 21, "E": 22, "D": 23,
                     "G": 24, "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, "O": 31, "N": 32, "P": 33, "S": 34,
                     "R": 35, "U": 36, "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, "]": 43, "_": 44, "a": 45,
                     "c": 46, "b": 47, "e": 48, "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, "l": 55, "o": 56,
                     "n": 57, "s": 58, "r": 59, "u": 60, "t": 61, "y": 62, "@": 63, "/": 64, "\\": 0}
    smiles_samples = pd.read_csv("./Luo/drug_smiles.csv", sep=',', header=None)
    smiles_sequences = []
    smile_max_len = 100
    for smile in smiles_samples[1]:
        smiles_sequences.append(seq_trans(smile, smile_max_len, CHARCANSMISET))
    smiles_sequences_array = np.array(smiles_sequences, dtype=np.int64)
    np.save('./predata/smiles_sequences.npy', smiles_sequences_array)
    CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                   "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23,
                   "X": 24, "Z": 25}
    protein_samples = pd.read_csv("./Luo/protein_fasta.csv", sep=',', header=None)
    protein_sequences = []
    pro_max_len = 1000
    for pro in protein_samples[1]:
        protein_sequences.append(seq_trans(pro, pro_max_len, CHARPROTSET))
    pro_sequences_array = np.array(protein_sequences, dtype=np.int64)
    np.save('./predata/pro_sequences.npy', pro_sequences_array)
    end_time = time.time()
    print('time cost:', end_time - start_time)


if __name__ == '__main__':
    preprocess()
