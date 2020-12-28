# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:57:23 2020

@author: Zhenqin Wu
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from queue import Queue
from sklearn.decomposition import PCA
import torch as t
import matplotlib
from torch.utils.data import Dataset
from models import PoincareEmbed, train
from labels import COMBINED_LABELS


eps = 1e-5
RAW_DATA_PATH = 'data/RAW/'
SAVE_DATA_PATH = 'data/'
DIST_INF = 100

class HierarchyDataLoader(Dataset):
    def __init__(self, 
                 X, 
                 y, 
                 y_order=None, 
                 n_same=8, 
                 n_pos=1, 
                 n_neg=10, 
                 sample_mode='tree',
                 **kwargs):
        self.X = X
        self.y = y
        self.y_order = y_order
        self.N = len(self.y)
        self.n_same = n_same
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.sample_mode = sample_mode
        
        self.classes = set(list(self.y))
        self.class_labels = set(c[-1] for c in self.classes)
        self.generate_dist_mapping()
        self.generate_class_sample_mapping()


    def generate_dist_mapping(self):
        self.dist_mapping = {}
        for c1 in self.classes:
            for c2 in self.classes:
                if (c1, c2) in self.dist_mapping:
                    continue
                dist = self.calculate_tree_dist(c1, c2)
                self.dist_mapping[(c1, c2)] = dist
                self.dist_mapping[(c2, c1)] = dist


    def generate_class_sample_mapping(self):
        self.class_mapping = {c: [] for c in self.classes}
        for i, _y in enumerate(self.y):
            self.class_mapping[_y].append(i)


    def calculate_tree_dist(self, c1, c2):
        shared_roots = 0
        for i in range(min(len(c1), len(c2))):
            if c1[i] == c2[i]:
                shared_roots += 1
            else:
                break
        if shared_roots == 0:
            return DIST_INF # Default for inf
        dist = 0
        for i in range(shared_roots, len(c1)):
            if c1[i] in self.class_labels:
                dist += 1
        for i in range(shared_roots, len(c2)):
            if c2[i] in self.class_labels:
                dist += 1
        return dist


    def __len__(self):
        return self.N
      
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()
        if self.sample_mode == 'tree':
            return self.__getitem_tree__(idx)
        elif self.sample_mode == 'linear':
            return self.__getitem_linear__(idx)
    
    def __getitem_linear__(self, idx):
        return self.X[idx], self.y_order[idx]

    def __getitem_tree__(self, idx):
        dist_classes = {c: self.dist_mapping[(self.y[idx], c)] for c in self.classes}
        dist_choice = sorted(set(dist_classes.values()))
        
        pos_sample_dist_ind = np.random.randint(0, len(dist_choice) - 1)
        assert dist_choice[pos_sample_dist_ind] < DIST_INF
        
        pos_samples = []
        neg_samples = []
        for c in self.classes:
            if dist_classes[c] == dist_choice[pos_sample_dist_ind]:
                pos_samples.extend(self.class_mapping[c])
            elif dist_classes[c] == dist_choice[pos_sample_dist_ind + 1]:
                neg_samples.extend(self.class_mapping[c])
            
        
        same_class_samples = np.random.choice(self.class_mapping[self.y[idx]], 
                                              (self.n_same,))
        pos_samples = np.random.choice(pos_samples, (self.n_pos,))
        neg_samples = np.random.choice(neg_samples, (self.n_neg,))
        
        batch_idx = np.concatenate([[idx], 
                                    same_class_samples, 
                                    pos_samples, 
                                    neg_samples])
        if self.y_order is None:
            return self.X[batch_idx]
        else:
            return self.X[batch_idx], np.array([self.y_order[_i] for _i in batch_idx])


    def embedding_dist_mat(self, embedding):
        if isinstance(embedding, t.Tensor):
            embedding = embedding.data.numpy()
        class_centers = {}
        for c in self.classes:
            inds = np.array(self.class_mapping[c])
            class_centers[c] = embedding[inds].mean(0)
        
        classes = sorted(self.classes)
        dist_mat = np.zeros((len(classes), len(classes)))
        for i, c1 in enumerate(classes):
            for j, c2 in enumerate(classes):
                dist_mat[i, j] = sample_poincare_dist(class_centers[c1],
                                                      class_centers[c2])
        return dist_mat


### Toy Datasets ###
def sample_poincare_dist(u, v):
    u = np.array(u).reshape((-1,))
    v = np.array(v).reshape((-1,))
    assert u.size == v.size

    diff_norm = np.square(np.linalg.norm(u - v, ord=2))
    u_norm = np.square(np.linalg.norm(u, ord=2))
    v_norm = np.square(np.linalg.norm(v, ord=2))
    
    dist = 1 + (2 * diff_norm / (1+1e-5 - u_norm) / (1+1e-5 - v_norm))
    dist = np.arccosh(dist)
    # dist = t.norm(u - v, dim=-1)
    return dist

def poincare_ratio(u, ratio=0.5):
    u = np.array(u).reshape((-1,))
    u_len = np.linalg.norm(u, ord=2)
    u_direction = u / u_len
    
    dist = sample_poincare_dist((0, 0), u)
    ratio_dist = dist * ratio
    
    x = np.cosh(ratio_dist) - 1
    new_len = np.sqrt(x / (2 + x))
    new_u = u_direction * new_len
    return new_u
    
    
    

def random_sampling(center, var=0.3, n_samples=100):
    assert center.shape[0] == 1
    n_dim = center.shape[1]
    X = np.random.normal(0, var, (n_samples, n_dim)) + center
    return X


def generate_toy_data(split=3, 
                      layers=2, 
                      samples_per_node=100,
                      n_dim=128,
                      seed=None):
    if not seed is None:
        np.random.seed(seed)
    X = []
    y = []
    nodes = {}
    q = Queue()  
    root_node_name = (0,)
    root_node = np.random.normal(0, 1, (1, n_dim))
    nodes[root_node_name] = root_node
    q.put(root_node_name)
    
    X.append(random_sampling(root_node, n_samples=samples_per_node))
    y.extend([root_node_name] * samples_per_node)
    
    while not q.empty():
        r_node_name = q.get()
        r_node = nodes[r_node_name]
        for i in range(split):
            random_mask = (np.random.rand(*r_node.shape) < 0.2) * 1
            child_node_name = r_node_name + (i,)
            child_node = r_node + np.random.normal(0, 0.3, r_node.shape) * random_mask
            nodes[child_node_name] = child_node
            if len(child_node_name) <= layers:
                q.put(child_node_name)
      
            X.append(random_sampling(child_node, n_samples=samples_per_node))
            y.extend([child_node_name] * samples_per_node)
    X = np.concatenate(X, 0)
    return X, y, nodes

    
### plate and droplet-seq datasets ###
def get_shared_gene_sets(paths, 
                         species=None,
                         gene_mapping_table='data/homologene.mouse2human.txt'):

    mapping = pd.read_table(gene_mapping_table)
    mapping = dict(zip(mapping['mouseGene'], mapping['humanGene']))
    gene_sets = []
    for i, path in enumerate(paths):
        df = pd.read_table(path)
        genes = [g.upper() for g in list(df[df.columns[0]])]
        if not species is None and species[i] != 'human':
            genes = [mapping[g].upper() for g in genes if g in mapping]
        gene_sets.append(genes)
        
    shared_genes = set(gene_sets[0])
    for gs in gene_sets:
        shared_genes = shared_genes & set(gs)
    shared_genes = sorted(shared_genes)
    return shared_genes
    
    
def read_data(path, 
              cytoTRACE_path=None, 
              use_gene_sets=None,
              map_to_human=False,
              gene_mapping_table='data/homologene.mouse2human.txt'):
    
    df = pd.read_table(path)
    genes = [g.upper() for g in list(df[df.columns[0]])]
    cell_ids = list(df.columns[1:])
    
    if map_to_human:
        mapping = pd.read_table(gene_mapping_table)
        mapping = dict(zip(mapping['mouseGene'], mapping['humanGene']))
        use_row_inds = [i for i, g in enumerate(genes) if g in mapping]
        df = df.iloc[np.array(use_row_inds)]
        genes = [mapping[genes[i]] for i in use_row_inds]
    
    if use_gene_sets:
        use_row_inds = [genes.index(g) for g in use_gene_sets]
        df = df.iloc[np.array(use_row_inds)]
        genes = [genes[i] for i in use_row_inds]
        assert set(genes) == set(use_gene_sets)
    
    X = np.transpose(np.array(df)[:, 1:]).astype(float)
    if cytoTRACE_path:
        df2 = pd.read_csv(cytoTRACE_path)
        vals = dict(zip(np.array(df2)[:, 0], np.array(df2)[:, 1:]))
        use_row_inds = [i for i, cell_id in enumerate(cell_ids) if cell_id in vals]
        if len(use_row_inds) < X.shape[0]:
            print("Cell name mismatching for %d samples" % (X.shape[0] - len(use_row_inds)))
        X = X[use_row_inds]
        cell_ids = [cell_ids[i] for i in use_row_inds]
        cytoTRACE_feats = np.stack([vals[i] for i in cell_ids], 0)
        genes.extend(list(df2.columns[1:]))
        return (X, cytoTRACE_feats), cell_ids, genes
    else:
        return (X,), cell_ids, genes
        


def normalize_order(orders, 
                    order_min=None, 
                    order_max=None, 
                    dist='euclidean',
                    dist_max=0.99):
    levels = [set(order) for order in orders]
    levels = set.union(*levels)
    levels = sorted(levels)
    if order_min is None:
        order_min = min(levels)
    if order_max is None:
        order_max = max(levels)
    
    order_interval = order_max - order_min
    level_mapping = {l: (l - order_min)/order_interval for l in levels}
    if dist == 'euclidean':
        level_mapping = {l: dist_max * level_mapping[l] for l in level_mapping}
    elif dist == 'poincare':
        level_mapping = {l: poincare_ratio((0, dist_max), ratio=level_mapping[l])[1]
                         for l in level_mapping}
    else:
        raise ValueError
    print(level_mapping)
    new_orders = [np.array([level_mapping[o] for o in order]) for order in orders]
    return new_orders
    

def read_label(path, use_order=False, use_cell_sets=None):
    df = pd.read_csv(path)
    cell_ids = list(df[df.columns[0]])
    if use_cell_sets:
        use_row_inds = [i for i, cell_id in enumerate(cell_ids) if cell_id in use_cell_sets]
        df = df.iloc[use_row_inds]
        cell_ids = list(df[df.columns[0]])
    phenotypes = df['phenotype']
    order = list(df['combined_order'])
    
    labels = []
    for p, o in zip(phenotypes, order):
        if p in COMBINED_LABELS:
            labels.append(COMBINED_LABELS[p][0])
            assert o == COMBINED_LABELS[p][1]
        else:
            labels.append(('Unknown',))
            print("Unknown phenotype %s" % p)
    if not use_order:
        return labels, cell_ids
    else:
        return labels, order, cell_ids


def normalize_as_rank(X, mode='sort'):
    n_rows = X.shape[0]
    output = []
    for r in np.arange(n_rows):
        if mode == 'unique':
            row_vals = sorted(np.unique(X[r]))
            n_vals = len(row_vals)
            if n_vals == 1:
                X[r] = 0
            else:
                val_mapping = {val: row_vals.index(val)/(n_vals - 1) \
                               for val in row_vals}
                new_row = np.array([val_mapping[v] for v in X[r]])
                output.append(new_row)
        elif mode == 'sort':
            new_row = np.zeros_like(X[r])
            prev_ind = None
            prev_val = None
            for ind, val in zip(np.argsort(X[r]), np.linspace(0, 1, len(new_row))):
                if prev_ind is None or X[r, ind] > X[r, prev_ind]:
                    new_row[ind] = val
                    prev_val = val
                else:
                    new_row[ind] = prev_val
                prev_ind = ind
            output.append(new_row)
    output = np.stack(output, 0)
    return output


if __name__ == '__main__':
    pass