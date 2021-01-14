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

normalization_method = 'rank'
# Define gene set
g_set = get_shared_gene_sets(data_paths, species)


# %% Read data

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

# %% Prepare for test data

datasets = [
    "Lung_fibroblast_C1",
    "Cortical_interneurons_C1",
    "Lung_development_C1",
    "Dentate_gyrus_timepoints_10x",
    "Hair_epidermis_C1",
    "Skeletal_stem_cells_C1",
    "Germ_cells_Smartseq2",
    "Hepatoblast_smart_seq2",
    "Thymus_Dropseq",
    "Peripheral_glia_Smart_seq2",
    "Blastocyst_phenotypes_SC3_seq",
    "Dentate_gyrus_phenotypes_10x",
    "Pancreatic_beta_cell_Smart_seq2",
    "Aging_HSCs_Smartseq2",
    "Neural_stem_cells_Dropseq",
    "mESC_invitro_ramDAseq",
    "Oligodendrocytes_timepoints_C1",
    "Lgr5CreER_intestine_CEL_seq",
    "Embryonic_HSCs_Tang_et_al",
    "Endometrium_CEL_seq",
    "HSMM_C1",
    "Invitro_NPCs_C1",
    "Oligodendrocytes_phenotypes_C1",
    "Pancreatic_alpha_cell_Smart_seq2",
    "hESC_in_vitro_C1",
    "Medial_ganglionic_eminence_C1",
    "Early_zebrafish_drop_seq",
    # "Whole_planaria_Dropseq",
]

species = [
    "Mouse",
    "Mouse",
    "Mouse",
    "Mouse",
    "Mouse",
    "Mouse",
    "Human",
    "Mouse",
    "Mouse",
    "Mouse",
    "Human", # Actually "Macaque"
    "Mouse",
    "Mouse",
    "Mouse",
    "Mouse",
    "Mouse",
    "Mouse",
    "Mouse",
    "Mouse",
    "Mouse",
    "Human",
    "Human",
    "Mouse",
    "Mouse",
    "Human",
    "Mouse",
    "Human", # Actually Zebrafish
    # "Planaria",
]

gene_mapping_table='data/homologene.mouse2human.txt'
mapping = pd.read_table(gene_mapping_table)
mapping = dict(zip(mapping['mouseGene'], mapping['humanGene']))

combined_data = []
for dataset, specie in zip(datasets, species):
    X_df = pd.read_csv("data/storeWebsiteDatasets/" + dataset + "_downsampled.tsv", sep='\t', index_col=0).T
    y = pd.read_csv("data/storeWebsiteDatasets/" + dataset + "_phenotypes.csv")
    assert X_df.shape[0] == y.shape[0]
    
    if specie.lower() not in ["mouse", "human"]:
        continue
    print(dataset)
    X = np.zeros((X_df.shape[0], len(g_set)))
    ct = 0
    for col in X_df.columns:
        try:
            if specie.lower() == "human":
                gene_name = col.upper()
            elif specie.lower() == "mouse":
                if col.upper() in mapping:
                    gene_name = mapping[col.upper()].upper()
                else:
                    continue
            if gene_name in g_set:
                ind = g_set.index(gene_name)
                X[:, ind] = np.array(X_df[col])
                ct += 1
        except Exception as e:
            print(e)
            print("Cannot parse gene %s" % str(col))
            
    print("Found %d/%d genes" % (ct, len(g_set)))
    
    y = np.array(y)
    valid_row_inds = np.where(y == y)[0]
    y = list(y[valid_row_inds][:, 0])
    X = X[valid_row_inds]
    
    cell_ids = [X_df.index[i] for i in valid_row_inds]
    genes = g_set
    y_order = y
        
    if normalization_method == 'sd':
        X = X / (1e-5 + X.std(0, keepdims=True)) # looking for better batch correction method
    elif normalization_method == 'rank':
        X = normalize_as_rank(X, mode='sort')
    print(X.shape)
    combined_data.append((X, 
                          y, 
                          y_order,
                          cell_ids, 
                          genes,
                          dataset))

# with open('temp_save_test_combined_rank.pkl', 'wb') as f:
#     pickle.dump(combined_data, f)