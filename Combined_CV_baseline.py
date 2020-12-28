0# -*- coding: utf-8 -*-
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
from models import PoincareEmbed, PoincareEmbedBaseline, MLP_pred, train, evaluate_embedding_with_xgboost, weighted_spearman_corr
from data import HierarchyDataLoader
from data import get_shared_gene_sets, read_data, read_label, normalize_as_rank, normalize_order
from plot import plot_embedding, plot_embedding_norm

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
use_cytoTRACE = True
model_root = 'model_save/cv-baseline-extra'

kwargs = {}
### DATASET SETTING ###
kwargs['sample_mode'] = 'linear'

### MODEL SETTING ###
kwargs['n_hidden'] = 256
kwargs['n_poincare'] = 8
kwargs['order_loss_mode'] = 'classification'
kwargs['gpu'] = True

### TRAIN SETTING ###
kwargs['lr'] = 1e-3
kwargs['batch_size'] = 128
kwargs['n_epochs'] = 1
kwargs['weight_decay'] = 0.01


excluded_phenotypes = ['Unknown',
                       # 'B1', 'B3', 'G2', 'G4', 'M1', 'M2', 'O1',
                       # 'Mp1', 'E2', 'E3', 'Mk1',
                       # 'N0', 'N1', 'N2',
                       # 'Endocrine', 'Goblet', 'I1', 'I2', 'I3', 'Paneth', 'Tuft',
                       ]


# %% Load Data
combined_data = pickle.load(open('./temp_save_combined_%s.pkl' % normalization_method, 'rb'))

Xs = []
ys = []
y_orders = []

for pair in combined_data:
    data_X, data_y, data_y_order, _, _, data_name = pair
    if not use_cytoTRACE:
        data_X = data_X[:, :-3] # Use only gene inputs
    data_y = list(data_y)
    data_y_order = list(data_y_order)
    row_inds = np.array([i for i, _y in enumerate(data_y) if \
                         _y[-1] not in excluded_phenotypes])
    n_excluded = len([_y for _y in data_y if _y[-1] in excluded_phenotypes])
    print("%s excludes %d samples" % (data_name, n_excluded))
    
    Xs.append(data_X[row_inds])
    ys.append([data_y[i] for i in row_inds])
    y_orders.append([data_y_order[i] for i in row_inds])

numeric_y_orders = normalize_order(y_orders, dist='euclidean')

# %% Cross Validation setup

for id_run in range(5):
    print("RUN%d" % id_run)
    test_preds = []
    test_trues = []
    spearman_scores = []
    for fold_i in range(len(data_paths)):
        data_name = os.path.split(data_paths[fold_i])[-1].split('_downsampled')[0]
        print(data_name)
        
        ### Setup datasets ###
        test_Xs = Xs[fold_i]
        test_ys = np.array(ys[fold_i])
        test_y_orders = np.array(numeric_y_orders[fold_i])
        
        train_Xs = np.concatenate([Xs[i] for i in range(len(data_paths)) if i != fold_i], 0)
        train_ys = np.concatenate([ys[i] for i in range(len(data_paths)) if i != fold_i], 0)
        train_y_orders = np.concatenate([numeric_y_orders[i] for i in range(len(data_paths)) if i != fold_i], 0)
        
        assert test_Xs.shape[0] == test_ys.shape[0] == test_y_orders.shape[0]
        assert train_Xs.shape[0] == train_ys.shape[0] == train_y_orders.shape[0]
        
        
        inds = np.arange(len(train_Xs))
        np.random.seed(123)
        np.random.shuffle(inds)
        train_inds = inds[int(0.2*len(inds)):]
        valid_inds = inds[:int(0.2*len(inds))]
        
        train_dataset = HierarchyDataLoader(train_Xs[train_inds].astype(float), 
                                            train_ys[train_inds], 
                                            y_order=train_y_orders[train_inds],
                                            **kwargs)
        valid_dataset = HierarchyDataLoader(train_Xs[valid_inds].astype(float), 
                                            train_ys[valid_inds], 
                                            y_order=train_y_orders[valid_inds],
                                            **kwargs)
        test_dataset = HierarchyDataLoader(test_Xs.astype(float), 
                                           test_ys, 
                                           y_order=test_y_orders, 
                                           **kwargs)
        
        print("Number of training samples: %d" % train_dataset.X.shape[0])
        print("Number of validation samples: %d" % valid_dataset.X.shape[0])
        print("Number of test samples: %d" % test_dataset.X.shape[0])

        ### Define Model ###
        # model = MLP_pred(n_dim=train_dataset.X.shape[1], **kwargs)
        model = PoincareEmbedBaseline(n_dim=train_dataset.X.shape[1], **kwargs)
        
        def evaluate(model):
            valid_preds = model.predict(valid_dataset)
            valid_trues = valid_dataset.y_order
            valid_score = weighted_spearman_corr(valid_preds, valid_trues)
            test_preds = model.predict(test_dataset)
            test_trues = test_dataset.y_order
            test_score = weighted_spearman_corr(test_preds, test_trues)
            print("Weighted spearman-r:\t%.3f\t%.3f" % (valid_score, test_score))
            return (valid_score, test_score)

        model_save = os.path.join(model_root, 'cv-baseline-classification-run%d/cv-baseline-%s' % (id_run, data_name))
        os.makedirs(model_save, exist_ok=True)
        
        ### Model training ###
        score_output_f = os.path.join(model_save, 'score_output.csv')
        with open(score_output_f, 'w') as f:
            pass
        
        for ct in range(30):
            model = train(model, train_dataset, **kwargs)
            t.save(model.state_dict(), os.path.join(model_save, 'save%d.pt' % ct))
            scores = list(evaluate(model))
            with open(score_output_f, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([ct] + scores)
        scores = np.array(pd.read_csv(score_output_f, header=None))
        argmax_ind = np.argmax(scores[:, 1])
        print(scores[argmax_ind])
        assert os.system("cp \"%s/save%d.pt\" \"%s/bkp.pt\"" % (model_save, scores[argmax_ind][0], model_save)) == 0

        ### Evaluations ###
        model.load(os.path.join(model_save, 'bkp.pt'))
        test_pred = model.predict(test_dataset)
        test_preds.append(test_pred)
        test_trues.append(test_dataset.y_order)
        spearman_scores.append(weighted_spearman_corr(test_pred, test_dataset.y_order))

    test_preds = np.concatenate(test_preds, 0)
    test_trues = np.concatenate(test_trues, 0)
    print("Mean scores")
    print(np.mean(spearman_scores))
    print("Overall score")
    print(weighted_spearman_corr(test_preds[:, 0], test_trues))

# %% EVALUATION

test_preds = [[] for _ in range(5)]
spearman_scores = [[] for _ in range(5)]

for fold_i in range(len(data_paths)):
    data_name = os.path.split(data_paths[fold_i])[-1].split('_downsampled')[0]
    print(data_name)
    
    ### Setup datasets ###
    test_Xs = Xs[fold_i]
    test_ys = np.array(ys[fold_i])
    test_y_orders = np.array(numeric_y_orders[fold_i])
    
    assert test_Xs.shape[0] == test_ys.shape[0] == test_y_orders.shape[0]

    test_dataset = HierarchyDataLoader(test_Xs.astype(float), 
                                       test_ys, 
                                       y_order=test_y_orders, 
                                       **kwargs)
    
    ### Define Model ###
    model = MLP_pred(n_dim=test_dataset.X.shape[1], **kwargs)
    # model = PoincareEmbedBaseline(n_dim=test_dataset.X.shape[1], **kwargs)
        
    def evaluate(model):
        test_preds = model.predict(test_dataset)
        test_trues = test_dataset.y_order
        test_score = weighted_spearman_corr(test_preds, test_trues)
        print("Weighted spearman-r:\t%.3f" % test_score)
        return test_score
    
    for i in range(5):
        model_save = os.path.join(model_root, 'cv-baseline-classification-run%d/cv-baseline-%s' % (i, data_name))
        score_df = np.array(pd.read_csv(os.path.join(model_save, 'score_output.csv'), header=None))
        best_model_idx = np.argmax(score_df[:, 1])
        best_model_path = os.path.join(model_save, "save%d.pt" % score_df[best_model_idx, 0])
        model.load_state_dict(t.load(best_model_path, map_location=lambda storage, loc: storage))
        test_pred = model.predict(test_dataset)
        test_preds[i].append(test_pred)
        spearman_scores[i].append(weighted_spearman_corr(test_pred, test_dataset.y_order))


print("Datasetname\tMean\tSD\tENSEMBLE")
spearman_scores = np.array(spearman_scores)
for fold_i in range(len(data_paths)):
    data_name = os.path.split(data_paths[fold_i])[-1].split('_downsampled')[0]
    m = np.mean(spearman_scores[:, fold_i])
    sd = np.std(spearman_scores[:, fold_i])
    ensemble_preds = np.concatenate([test_preds[i][fold_i] for i in range(5)], 1).mean(1)
    ensemble_corr = weighted_spearman_corr(ensemble_preds, numeric_y_orders[fold_i])
    print("%s\t%.3f\t%.3f\t%.3f" % 
          (data_name, 
           m, 
           sd, 
           ensemble_corr))
    
    # plt.clf()
    # # test_dataset = HierarchyDataLoader(Xs[fold_i], 
    # #                                    ys[fold_i], 
    # #                                    y_order=y_orders[fold_i],
    # #                                    **kwargs)
    # plot_embedding_norm(ensemble_preds, y_orders[fold_i], renorm=False)
    # plt.savefig("violinplot_tree_%s.png" % data_name, dpi=300)

combined_test_preds = [np.concatenate(pred)[:, 0] for pred in test_preds]
ensemble_preds = np.stack(combined_test_preds, 1).mean(1)
test_trues = np.concatenate(numeric_y_orders)
overall_scores = [weighted_spearman_corr(pred, test_trues) for pred in combined_test_preds]
ensemble_score = weighted_spearman_corr(ensemble_preds, test_trues)

# plt.clf()
# plot_embedding_norm(ensemble_preds, np.concatenate(y_orders), renorm=False)
# plt.savefig("violinplot_baseline_all.png", dpi=300)

print()
print("Overall\t%.3f\t%.3f\t%.3f" % 
      (np.mean(overall_scores), 
       np.std(overall_scores), 
       ensemble_score))
