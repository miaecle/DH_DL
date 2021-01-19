# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:41:35 2021

@author: Zhenqin Wu
"""
import os
import numpy as np
import csv
import pickle
import matplotlib
import matplotlib.pyplot as plt
import scipy
import argparse
import pandas as pd
import time
from sklearn.decomposition import PCA
import torch as t
from functools import partial
from torch.utils.data import Dataset
from models import PoincareEmbed, PoincareEmbedBaseline, MLP_pred, train, evaluate_embedding_with_xgboost, weighted_spearman_corr
from data import HierarchyDataLoader
from data import get_shared_gene_sets, read_data, read_label, normalize_as_rank, normalize_order
from plot import plot_embedding, plot_embedding_norm
from smooth import euclidean_similarity, cosine_similarity, find_kNN, get_embeddings, get_preds, get_preds_from_embeddings, smoothing_by_average_kNN


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


def main(args):
    normalization_method = 'rank'
    use_cytoTRACE = False
    model_root = args.model_root
    output_path = args.output_path

    kwargs = {}
    ### DATASET SETTING ###
    kwargs['n_same'] = 0
    kwargs['n_pos'] = 1
    kwargs['n_neg'] = 10
    kwargs['sample_mode'] = 'tree'

    ### MODEL SETTING ###
    kwargs['loss_type'] = 'energy'
    kwargs['dist'] = 'poincare'
    kwargs['n_hidden'] = 256
    kwargs['n_poincare'] = 8
    kwargs['tree_loss_weight'] = 0.1
    kwargs['node_loss_weight'] = 0.
    kwargs['order_loss_weight'] = 1.
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
    Xs = []
    ys = []
    y_orders = []
    data_names = []

    if args.test:
        combined_data = pickle.load(open('temp_save_test_combined_%s.pkl' % normalization_method, 'rb'))
        for pair in combined_data:
            data_X, data_y, data_y_order, _, _, data_name = pair
            Xs.append(data_X)
            ys.append(np.array(data_y))
            y_orders.append(np.array(data_y_order))
            data_names.append(data_name)
        numeric_y_orders = y_orders

    else:
        combined_data = pickle.load(open('temp_save_combined_%s.pkl' % normalization_method, 'rb'))
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
            data_names.append(data_name)
        numeric_y_orders = normalize_order(y_orders, dist='euclidean')

    # %%
    def evaluate(use_pool='train',
                 neighbor_mode='feature', 
                 average_mode='feature', 
                 k=20, 
                 sim_fn=cosine_similarity):
        
        num_runs = 5
        test_preds = [[] for _ in range(num_runs)]
        spearman_scores = [[] for _ in range(num_runs)]
        
        for fold_i in range(len(data_names)):
            data_name = data_names[fold_i].split('_downsampled')[0]
            
            ### Setup datasets ###
            test_Xs = Xs[fold_i].astype(float)
            test_ys = np.array(ys[fold_i])
            test_y_orders = np.array(numeric_y_orders[fold_i])
            assert test_Xs.shape[0] == test_ys.shape[0] == test_y_orders.shape[0]
        
            train_Xs = np.concatenate([X for _i, X in enumerate(Xs) if _i != fold_i], 0).astype(float)

            if use_pool == 'train':
                pool_Xs = train_Xs
            elif use_pool == 'test':
                pool_Xs = test_Xs
            elif use_pool == 'train+test':
                pool_Xs = np.concatenate([train_Xs, test_Xs], 0)
            
            kNN_inds = None
            if neighbor_mode == 'feature':
                kNN_inds = find_kNN(test_Xs, pool_Xs, k=k, sim_fn=sim_fn)
            
            ### Define Model ###
            model = PoincareEmbed(n_dim=test_Xs.shape[1], **kwargs)
            
            for i in range(num_runs):
                model_save = os.path.join(model_root, 'cv-tree-classification-run%d/cv-tree-%s' % (i, data_name))
                score_df = np.array(pd.read_csv(os.path.join(model_save, 'score_output.csv'), header=None))
                best_model_idx = np.argmax(score_df[:, 1])
                best_model_path = os.path.join(model_save, "save%d.pt" % score_df[best_model_idx, 0])
                model.load_state_dict(t.load(best_model_path, map_location=lambda storage, loc: storage))
        
                test_embeddings = get_embeddings(test_Xs, model)
                pool_embeddings = get_embeddings(pool_Xs, model)
                if neighbor_mode == 'embedding':
                    kNN_inds = find_kNN(test_embeddings, pool_embeddings, k=k, sim_fn=sim_fn)

                test_pred = None
                assert kNN_inds is not None
                if average_mode == 'feature':
                    _test_Xs = smoothing_by_average_kNN(pool_Xs, kNN_inds)
                    test_pred = get_preds(_test_Xs, model)
                elif average_mode == 'embedding':
                    _test_embs = smoothing_by_average_kNN(pool_embeddings, kNN_inds)
                    test_pred = get_preds_from_embeddings(_test_embs, model)
                elif average_mode == 'label':
                    pool_preds = get_preds_from_embeddings(pool_embeddings, model)
                    test_pred = smoothing_by_average_kNN(pool_preds, kNN_inds)

                test_preds[i].append(test_pred)
                spearman_scores[i].append(weighted_spearman_corr(test_pred, test_y_orders))
        
            m = np.mean([spearman_scores[i][fold_i] for i in range(num_runs)])
            sd = np.std([spearman_scores[i][fold_i] for i in range(num_runs)])
            ensemble_preds = np.concatenate([test_preds[i][fold_i] for i in range(num_runs)], 1).mean(1)
            ensemble_corr = weighted_spearman_corr(ensemble_preds, test_y_orders)
            with open(output_path, 'a') as f:
                f.write("%s\t%.3f\t%.3f\t%.3f\n" % (data_name, m, sd, ensemble_corr))
        
        combined_test_preds = [np.concatenate(pred)[:, 0] for pred in test_preds]
        ensemble_preds = np.stack(combined_test_preds, 1).mean(1)
        test_trues = np.concatenate(numeric_y_orders)
        overall_scores = [weighted_spearman_corr(pred, test_trues) for pred in combined_test_preds]
        ensemble_score = weighted_spearman_corr(ensemble_preds, test_trues)
        
        with open(output_path, 'a') as f:
            f.write("\nOverall\t%.3f\t%.3f\t%.3f\n" % (np.mean(overall_scores), 
                                                       np.std(overall_scores), 
                                                       ensemble_score))



    # %%
    def evaluate_test(use_pool='test',
                      neighbor_mode='feature', 
                      average_mode='feature', 
                      k=20, 
                      sim_fn=cosine_similarity):
        
        test_preds = [[] for _ in range(len(data_names))]
        spearman_scores = [[] for _ in range(len(data_names))]
        
        for fold_i in range(len(data_names)):
            data_name = data_names[fold_i].split('_downsampled')[0]
            
            ### Setup datasets ###
            test_Xs = Xs[fold_i].astype(float)
            test_ys = np.array(ys[fold_i])
            test_y_orders = np.array(numeric_y_orders[fold_i])
            assert test_Xs.shape[0] == test_ys.shape[0] == test_y_orders.shape[0]
        
            if use_pool == 'test':
                pool_data = test_Xs
            else:
                raise ValueError("pool data not found")
            
            kNN_inds = None
            if neighbor_mode == 'feature':
                kNN_inds = find_kNN(test_Xs, pool_Xs, k=k, sim_fn=sim_fn)
            
            ### Define Model ###
            model = PoincareEmbed(n_dim=test_Xs.shape[1], **kwargs)
            
            for i in range(5):
                for valid_data_name in data_paths:
                    valid_data_name = os.path.split(valid_data_name)[-1].split("_downsampled")[0]
                    model_save = os.path.join(model_root, 'cv-tree-classification-run%d/cv-tree-%s' % (i, valid_data_name))
                    score_df = np.array(pd.read_csv(os.path.join(model_save, 'score_output.csv'), header=None))
                    best_model_idx = np.argmax(score_df[:, 2])
                    best_model_path = os.path.join(model_save, "save%d.pt" % score_df[best_model_idx, 0])
                    model.load_state_dict(t.load(best_model_path, map_location=lambda storage, loc: storage))

                    test_embeddings = get_embeddings(test_Xs, model)
                    pool_embeddings = get_embeddings(pool_Xs, model)
                    if neighbor_mode == 'embedding':
                        kNN_inds = find_kNN(test_embeddings, pool_embeddings, k=k, sim_fn=sim_fn)

                    test_pred = None
                    assert kNN_inds is not None
                    if average_mode == 'feature':
                        _test_Xs = smoothing_by_average_kNN(pool_Xs, kNN_inds)
                        test_pred = get_preds(_test_Xs, model)
                    elif average_mode == 'embedding':
                        _test_embs = smoothing_by_average_kNN(pool_embeddings, kNN_inds)
                        test_pred = get_preds_from_embeddings(_test_embs, model)
                    elif average_mode == 'label':
                        pool_preds = get_preds_from_embeddings(pool_embeddings, model)
                        test_pred = smoothing_by_average_kNN(pool_preds, kNN_inds)

                    test_preds[fold_i].append(test_pred)
                    spearman_scores[fold_i].append(weighted_spearman_corr(test_pred, test_y_orders))
        
        
            m = np.mean(spearman_scores[fold_i])
            sd = np.std(spearman_scores[fold_i])
            ensemble_preds = np.concatenate(test_preds[fold_i], 1).mean(1)
            ensemble_corr = weighted_spearman_corr(ensemble_preds, test_y_orders)
            print("%s\t%.3f\t%.3f\t%.3f\n" % (data_name, m, sd, ensemble_corr))
            with open(output_path, 'a') as f:
                f.write("%s\t%.3f\t%.3f\t%.3f\n" % (data_name, m, sd, ensemble_corr))


    # %% RUN
    for use_pool in ['train', 'test']:
      for neighbor_mode in ['feature', 'embedding']:
        for average_mode in ['feature', 'embedding', 'label']:
            sim_fn = cosine_similarity
            for k in [10, 40]:
                if neighbor_mode == 'feature' and 'train' in use_pool:
                    # skipping
                    continue
                with open(output_path, 'a') as f:
                    f.write("%s_%s_%s_%d_%s\n" % (use_pool, neighbor_mode, average_mode, k, sim_fn.__name__))

                t1 = time.time()
                if args.test:
                    evaluate_test(use_pool=use_pool,
                                  neighbor_mode=neighbor_mode, 
                                  average_mode=average_mode, 
                                  k=k, 
                                  sim_fn=sim_fn)
                else:
                    evaluate(use_pool=use_pool,
                             neighbor_mode=neighbor_mode, 
                             average_mode=average_mode, 
                             k=k, 
                             sim_fn=sim_fn)
                t2 = time.time()

                with open(output_path, 'a') as f:
                    f.write("%.3f\n" % (t2 - t1))





def parse_args():
    parser = argparse.ArgumentParser(description='testing on smoothing inputs')

    # Input-output shape
    parser.add_argument('--model_root',
                        type=str,
                        default='',
                        help='root directory of model')
    parser.add_argument('--output_path',
                        type=str,
                        default="",
                        help="output txt path")
    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        help="evaluate on test data")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)
