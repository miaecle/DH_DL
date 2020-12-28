# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:23:43 2020

@author: Zhenqin Wu
"""

import numpy as np
import csv
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
# test_data_paths = ['data/plateOntogenyDatasets/MACA_marrow_downsampled_with_ontogeny.tsv']
# test_label_paths = ['data/plateOntogenyDatasets/MACA_marrow_phenotype.csv']
test_data_paths = ['data/bone_marrow/batch_corrected_bone_marrow/bone_marrow_smart-seq2_batch_corrected_with_ontogeny.tsv']
test_label_paths = ['data/bone_marrow/batch_corrected_bone_marrow/bone_marrow_smart-seq2_batch_corrected_phenotype.csv']

# train_data_paths = ['data/dropletOntogenyDatasets/Mouse_Data_Marrow_10x_MACA_downsampled_with_ontogeny.tsv']
# train_label_paths = ['data/dropletOntogenyDatasets/Mouse_Data_Marrow_10x_MACA_phenotype.csv']
train_data_paths = ['data/bone_marrow/batch_corrected_bone_marrow/bone_marrow_10x_batch_corrected_with_ontogeny.tsv']
train_label_paths = ['data/bone_marrow/batch_corrected_bone_marrow/bone_marrow_10x_batch_corrected_phenotype.csv']

normalization_method = None

loss_type = 'energy'
dist = 'poincare'
use_order = True

# %% Load data

# Define gene set
g_set = get_shared_gene_sets(train_data_paths + test_data_paths)

train_data = []
for data_path, label_path in zip(train_data_paths, train_label_paths):
    X, cell_ids1, genes = read_data(data_path, use_gene_sets=g_set)
    y, y_order, cell_ids2 = read_label(label_path, use_order=True)
    assert cell_ids1 == cell_ids2
    if normalization_method == 'sd':
        X = X / (1e-5 + X.std(0, keepdims=True)) # looking for better batch correction method
    elif normalization_method == 'rank':
        X = normalize_as_rank(X, mode='sort')
    train_data.append((X, y, y_order, cell_ids1, genes))

test_data = []
for data_path, label_path in zip(test_data_paths, test_label_paths):
    X, cell_ids1, genes = read_data(data_path, use_gene_sets=g_set)
    y, y_order, cell_ids2 = read_label(label_path, use_order=True)
    assert cell_ids1 == cell_ids2
    if normalization_method == 'sd':
        X = X / (1e-5 + X.std(0, keepdims=True)) # looking for better batch correction method
    elif normalization_method == 'rank':
        X = normalize_as_rank(X, mode='sort')
    test_data.append((X, y, y_order, cell_ids1, genes))


# %% Filter data
print("Train Phenotype")
for pair in train_data:
    print(sorted(set(phe[-1] for phe in pair[1])))
print("Test Phenotype")
for pair in test_data:
    print(sorted(set(phe[-1] for phe in pair[1])))

excluded_phenotypes = []#'Mp1', 'E2', 'E3', 'Mk1']

train_Xs = []
train_ys = []
train_y_orders = []
test_Xs = []
test_ys = []
test_y_orders = []
for pair in train_data:
    row_inds = np.array([i for i, _y in enumerate(pair[1]) if \
                         _y[-1] not in excluded_phenotypes])
    print(len([_y for _y in pair[1] if _y[-1] in excluded_phenotypes]))
    train_ys.extend([pair[1][i] for i in row_inds])
    train_y_orders.extend([pair[2][i] for i in row_inds])
    train_Xs.append(pair[0][row_inds])
for pair in test_data:
    row_inds = np.array([i for i, _y in enumerate(pair[1]) if \
                         _y[-1] not in excluded_phenotypes])
    print(len([_y for _y in pair[1] if _y[-1] in excluded_phenotypes]))
    test_ys.extend([pair[1][i] for i in row_inds])
    test_y_orders.extend([pair[2][i] for i in row_inds])
    test_Xs.append(pair[0][row_inds])
train_Xs = np.concatenate(train_Xs, 0)
train_ys = np.array(train_ys)
train_y_orders = np.array(train_y_orders)
test_Xs = np.concatenate(test_Xs, 0)
test_ys = np.array(test_ys)
test_y_orders = np.array(test_y_orders)

inds = np.arange(len(train_Xs))
np.random.seed(123)
np.random.shuffle(inds)
train_inds = inds[int(0.2*len(inds)):]
valid_inds = inds[:int(0.2*len(inds))]

train_y_num_orders, test_y_num_orders = normalize_order([train_y_orders, test_y_orders], dist='poincare')
train_dataset = HierarchyDataLoader(train_Xs[train_inds], train_ys[train_inds], train_y_num_orders[train_inds])
valid_dataset = HierarchyDataLoader(train_Xs[valid_inds], train_ys[valid_inds], train_y_num_orders[valid_inds])
test_dataset = HierarchyDataLoader(test_Xs, test_ys, test_y_num_orders)

print("Number of training samples: %d" % train_dataset.X.shape[0])
print("Number of validation samples: %d" % valid_dataset.X.shape[0])
print("Number of test samples: %d" % test_dataset.X.shape[0])

# %% Define model
model = PoincareEmbed(n_dim=train_dataset.X.shape[1],
                      n_hidden=32,
                      n_poincare=2,
                      n_pos=1,
                      n_neg=10,
                      dist=dist,
                      loss_type=loss_type,
                      use_order=use_order,
                      gpu=False)

# model.load_state_dict(t.load('./model_save/bone_marrow_cross_batch_smartseq-to-10x_uncorrected_contrastive_margin2.pt', map_location=lambda storage, loc: storage))
def evaluate(model, evaluate_fn=evaluate_embedding_with_xgboost):
    scores = []
    for _ in range(10):
        train_preds = model.predict(train_dataset)
        valid_preds = model.predict(valid_dataset)
        test_preds = model.predict(test_dataset)
        train_scores, xgb_model = evaluate_fn(
            train_preds, train_dataset, return_model=True)
        valid_scores = evaluate_fn(valid_preds, valid_dataset, model=xgb_model)
        test_scores = evaluate_fn(test_preds, test_dataset, model=xgb_model)
        
        train_corr = weighted_spearman_corr(
            np.linalg.norm(train_preds, axis=1), train_dataset.y_order)
        valid_corr = weighted_spearman_corr(
            np.linalg.norm(valid_preds, axis=1), valid_dataset.y_order)
        test_corr = weighted_spearman_corr(
            np.linalg.norm(test_preds, axis=1), test_dataset.y_order)
        
        scores.append([train_scores[1], valid_scores[1], test_scores[1],
                       train_corr, valid_corr, test_corr])
    scores = np.array(scores).mean(0)
    print("V-measure:\t%.3f\t%.3f\t%.3f" % tuple(scores[:3]))
    print("Weighted spearman-r:\t%.3f\t%.3f\t%.3f" % tuple(scores[3:]))
    return scores


# %% Model training
score_output_f = 'model_save/score_output.csv'
with open(score_output_f, 'w') as f:
    pass

for ct in range(30):
    model = train(model, 
                  train_dataset,
                  lr=1e-3, 
                  batch_size=128, 
                  n_epochs=10,
                  weight_decay=0.01)
    t.save(model.state_dict(), 'model_save/save%d.pt' % ct)
    scores = list(evaluate(model))
    print(scores)
    with open(score_output_f, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([ct] + scores)
scores = np.array(pd.read_csv(score_output_f, header=None))
argmax_ind = np.argmax(scores[:, 2])
print(scores[argmax_ind])

# %% Plottings
dataset = train_dataset
embedding = model.predict(dataset)

plot_embedding(embedding, dataset, full_view=True, legend=True)
plt.savefig('full.png', dpi=300)
plot_embedding(embedding, dataset, renorm=True, with_unit_circle=False, legend=True)
plt.savefig('full_renorm.png', dpi=300)
plot_embedding_norm(embedding, dataset, renorm=True)
plt.savefig('full_renom_1d.png', dpi=300)

