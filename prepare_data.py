# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 00:18:05 2020

@author: Zhenqin Wu
"""
import os
import numpy as np
import csv
import pickle
import matplotlib
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from sklearn.decomposition import PCA
import torch as t
from functools import partial
from torch.utils.data import Dataset
from models import PoincareEmbed, train, evaluate_embedding_with_xgboost, weighted_spearman_corr
from data import HierarchyDataLoader
from data import get_shared_gene_sets, read_data, read_label, normalize_as_rank, normalize_order
from plot import plot_embedding, plot_embedding_norm


# %% Define data detail
data_paths = [
    'data/plateOntogenyDatasets/dendriticcell_counts_downsampled_with_ontogeny.tsv',
    'data/plateOntogenyDatasets/EPI_CountsTable_downsampled_with_ontogeny.tsv',
    'data/plateOntogenyDatasets/MACA_marrow_downsampled_with_ontogeny.tsv',
    'data/plateOntogenyDatasets/mouseembryo_counts_downsampled_with_ontogeny.tsv',
    'data/plateOntogenyDatasets/olsson_counts_downsampled_with_ontogeny.tsv',
    'data/plateOntogenyDatasets/HumanEmbryo_CountTable_New_downsampled_with_ontogeny.tsv',
    'data/plateOntogenyDatasets/Kyle_CountTable_New_downsampled_with_ontogeny.tsv',
    'data/dropletOntogenyDatasets/Mouse_Data_Marrow_10x_MACA_downsampled_with_ontogeny.tsv',
    'data/dropletOntogenyDatasets/Mouse_Data_DirectProtocol_Neuron_downsampled_with_ontogeny.tsv',
    'data/dropletOntogenyDatasets/Mouse_Data_StandardProtocol_Neuron_downsampled_with_ontogeny.tsv',
    'data/dropletOntogenyDatasets/Mouse_Data_Regev_SmartSeq.log2.TPM.Capitalized_downsampled_with_ontogeny.tsv',
    'data/dropletOntogenyDatasets/Intestine_Dropseq_downsampled_with_ontogeny.tsv',
    'data/dropletOntogenyDatasets/Human_Data_Sample_Blood_AllMerged_downsampled_with_ontogeny.tsv',
    ]

cytoTRACE_score_paths = [
    'data/xgboost_scores/dendriticcell_counts_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/EPI_CountsTable_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/MACA_marrow_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/mouseembryo_counts_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/olsson_counts_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/HumanEmbryo_CountTable_New_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/Kyle_CountTable_New_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/Mouse_Data_Marrow_10x_MACA_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/Mouse_Data_DirectProtocol_Neuron_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/Mouse_Data_StandardProtocol_Neuron_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/Mouse_Data_Regev_SmartSeq.log2.TPM.Capitalized_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/Intestine_Dropseq_cytotrace_xgboost_results.csv',
    'data/xgboost_scores/Human_Data_Sample_Blood_AllMerged_cytotrace_xgboost_results.csv',
    ]

label_paths = [
    'data/plateOntogenyDatasets/dendriticcell_counts_phenotype.csv',
    'data/plateOntogenyDatasets/EPI_CountsTable_phenotype.csv',
    'data/plateOntogenyDatasets/MACA_marrow_phenotype.csv',
    'data/plateOntogenyDatasets/mouseembryo_counts_phenotype.csv',
    'data/plateOntogenyDatasets/olsson_counts_phenotype.csv',
    'data/plateOntogenyDatasets/HumanEmbryo_CountTable_New_phenotype.csv',
    'data/plateOntogenyDatasets/Kyle_CountTable_New_phenotype.csv',
    'data/dropletOntogenyDatasets/Mouse_Data_Marrow_10x_MACA_phenotype.csv',
    'data/dropletOntogenyDatasets/Mouse_Data_DirectProtocol_Neuron_phenotype.csv',
    'data/dropletOntogenyDatasets/Mouse_Data_StandardProtocol_Neuron_phenotype.csv',
    'data/dropletOntogenyDatasets/Mouse_Data_Regev_SmartSeq.log2.TPM.Capitalized_phenotype.csv',
    'data/dropletOntogenyDatasets/Intestine_Dropseq_phenotype.csv',
    'data/dropletOntogenyDatasets/Human_Data_Sample_Blood_AllMerged_phenotype.csv',
    ]

species = ['mouse', 'mouse', 'mouse', 'mouse', 'mouse', 'human', 
           'human', 'mouse', 'mouse', 'mouse', 'mouse', 'mouse',
           'human']

test_data_paths = [
    # 'data/plateOntogenyDatasets/MACA_marrow_downsampled_with_ontogeny.tsv',
    'data/dropletOntogenyDatasets/Mouse_Data_Marrow_10x_MACA_downsampled_with_ontogeny.tsv',
    ]
test_label_paths = [
    # 'data/plateOntogenyDatasets/MACA_marrow_phenotype.csv',
    'data/dropletOntogenyDatasets/Mouse_Data_Marrow_10x_MACA_phenotype.csv',
    ]

normalization_method = 'rank'

n_same = 8
n_pos = 1
n_neg = 10

loss_type = 'contrastive'
dist = 'poincare'
gene_subset = False
use_order = True
node_loss_weight = 0.
order_loss_weight = 0.1

# %% Read data
# Define gene set
g_set = get_shared_gene_sets(data_paths, species)

# Load data
combined_data = []
for data_path, cytoTRACE_path, label_path, specie in \
    zip(data_paths, cytoTRACE_score_paths, label_paths, species):
    (X, cytoTRACE_feats), cell_ids1, genes = read_data(
        data_path, 
        cytoTRACE_path=cytoTRACE_path,
        use_gene_sets=g_set, 
        map_to_human=(specie != 'human'))
    y, y_order, cell_ids2 = read_label(label_path, use_order=True, use_cell_sets=cell_ids1)
    assert cell_ids1 == cell_ids2
    if normalization_method == 'sd':
        X = X / (1e-5 + X.std(0, keepdims=True)) # looking for better batch correction method
    elif normalization_method == 'rank':
        X = normalize_as_rank(X, mode='sort')
    print(X.shape)
    X = np.concatenate([X, cytoTRACE_feats], 1)
    print(X.shape)
    combined_data.append((X, 
                          y, 
                          y_order,
                          cell_ids1, 
                          genes,
                          os.path.split(data_path)[-1]))

with open('./temp_save_combined_%s.pkl' % normalization_method, 'wb') as f:
    pickle.dump(combined_data, f)

combined_data = pickle.load(open('./temp_save_combined_%s.pkl' % normalization_method, 'rb'))
