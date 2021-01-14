# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:16:22 2020

@author: Zhenqin Wu
"""

import numpy as np
import matplotlib.pyplot as plt
import xgboost
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import v_measure_score
from scipy.stats import spearmanr
import collections
import scipy

eps = 1e-5

class PoincareEmbed(nn.Module):
    def __init__(self,
                 n_dim=128,
                 n_hidden=32,
                 n_poincare=2,
                 n_same=8,
                 n_pos=1,
                 n_neg=10,
                 gpu=False,
                 dist='poincare',
                 loss_type='energy',
                 tree_loss_weight=1.0,
                 node_loss_weight=1.0,
                 order_loss_weight=1.0,
                 order_loss_mode='classification',
                 **kwargs):

        super(PoincareEmbed, self).__init__()
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.n_poincare = n_poincare
        self.gpu = gpu
        self.n_same = n_same
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.enc = nn.Sequential(
            nn.Linear(self.n_dim, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU()
            )

        self.poincare_embed_norm = nn.Sequential(
            nn.Linear(self.n_hidden, 1),
            nn.Sigmoid())
        
        self.poincare_embed_direction = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_poincare))
        
        self.pred_head = nn.Sequential(
            nn.Linear(self.n_poincare, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, 1)
            )
        
        if self.gpu:
            self.cuda()
        
        if dist == 'poincare':
            self.dist_fn = self.batch_poincare_dist
        elif dist == 'euclidean':
            self.dist_fn = self.batch_euclidean_dist
        else:
            raise ValueError
            
        if loss_type == 'energy':
            self.loss_fn = self.one_vs_rest_energy
        elif loss_type == 'contrastive':
            self.loss_fn = self.one_vs_rest_contrastive
        else:
            raise ValueError
        
        if order_loss_mode == 'classification':
            self.order_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif order_loss_mode == 'regression':
            self.order_loss_fn = nn.MSELoss(reduction='none')
            
        self.tree_loss_weight = tree_loss_weight
        self.node_loss_weight = node_loss_weight
        self.order_loss_weight = order_loss_weight
    
    @staticmethod
    def batch_poincare_dist(u, v):
        # shape of u: (X, ..., X, n_u, n_dim)
        # shape of v: (X, ..., X, n_v, n_dim)
        # output shape: (X, ..., X, n_u, n_v)
        
        assert u.shape[-1] == v.shape[-1]
        assert len(u.shape) == len(v.shape)
        dim = len(u.shape)
        u = u.unsqueeze(dim - 1)
        v = v.unsqueeze(dim - 2)
        diff_norm = t.square(t.norm(u - v, dim=-1))
        u_norm = t.square(t.norm(u, dim=-1))
        v_norm = t.square(t.norm(v, dim=-1))
        
        dist = 1 + (2 * diff_norm / (1 - u_norm).clamp(min=eps) / (1 - v_norm).clamp(min=eps)).clamp(min=eps)
        dist = t.acosh(dist)
        return dist
    
    @staticmethod
    def batch_euclidean_dist(u, v):
        # shape of u: (X, ..., X, n_u, n_dim)
        # shape of v: (X, ..., X, n_v, n_dim)
        # output shape: (X, ..., X, n_u, n_v)
        
        assert u.shape[-1] == v.shape[-1]
        assert len(u.shape) == len(v.shape)
        dim = len(u.shape)
        u = u.unsqueeze(dim - 1)
        v = v.unsqueeze(dim - 2)
        
        dist = t.norm(u - v, dim=-1)
        return dist


    def forward(self, inputs):
        _x = self.enc(inputs)
        norm = self.poincare_embed_norm(_x)
        direction = self.poincare_embed_direction(_x)
        direction_normed = direction / t.norm(direction, dim=-1, keepdim=True).clamp(min=eps)
        embedding = direction_normed * norm
        final_norm_pred = self.pred_head(embedding)
        
        return embedding, final_norm_pred


    def train_loss(self, inputs, order_labels=None):
        loss_dict = {}
        loss = 0
        
        embedding, norm = self.forward(inputs)
        
        query_embedding = embedding[:, 0:1]
        same_class_samples_embedding = embedding[:, 1:(self.n_same + 1)]
        pos_samples_embedding = embedding[:, (self.n_same + 1):(self.n_same + self.n_pos + 1)]
        neg_samples_embedding = embedding[:, (self.n_same + self.n_pos + 1):]
        
        tree_loss = self.loss_fn(query_embedding, pos_samples_embedding, neg_samples_embedding)
        loss_dict['tree_loss'] = tree_loss.sum() * self.tree_loss_weight
        loss += tree_loss.sum() * self.tree_loss_weight
        if self.n_same > 0:
            node_loss = self.homogeneity_loss(query_embedding, same_class_samples_embedding)
            loss_dict['node_loss'] = node_loss.sum() * self.node_loss_weight
            loss += node_loss.sum() * self.node_loss_weight
        
        assert order_labels is not None
        weights = t.ones_like(order_labels)
        weights[:, 1:] = 1 / (self.n_neg + self.n_pos)
        order_loss = self.order_loss_fn(norm.view(*order_labels.shape), order_labels)
        order_loss = (order_loss * weights).sum()
        loss_dict['order_loss'] = order_loss * self.order_loss_weight
        loss += order_loss * self.order_loss_weight
        return loss, loss_dict
        
        
    def predict(self, dataset):
        if isinstance(dataset, np.ndarray):
            inputs = t.from_numpy(dataset).float()
        else:
            inputs = t.from_numpy(dataset.X).float()
        if self.gpu:
            inputs = inputs.cuda()
        return self.forward(inputs)[1].cpu().data.numpy()


    def one_vs_rest_energy(self, 
                           query_embedding, 
                           pos_samples_embedding,
                           neg_samples_embedding,
                           temp=1.0):
        assert query_embedding.shape[1] == 1
        assert pos_samples_embedding.shape[1] == self.n_pos
        assert neg_samples_embedding.shape[1] == self.n_neg
        
        pos_dist = self.dist_fn(query_embedding, pos_samples_embedding)
        neg_dist = self.dist_fn(query_embedding, neg_samples_embedding)
        
        dist = t.cat([pos_dist, neg_dist], -1)
        logit = - dist / temp
        loss = t.logsumexp(logit, -1) - logit[..., 0]
        return loss


    def one_vs_rest_contrastive(self, 
                                query_embedding, 
                                pos_samples_embedding,
                                neg_samples_embedding,
                                m=2):
        assert query_embedding.shape[1] == 1
        assert pos_samples_embedding.shape[1] == self.n_pos
        assert neg_samples_embedding.shape[1] == self.n_neg
        
        pos_dist = self.dist_fn(query_embedding, pos_samples_embedding)
        neg_dist = self.dist_fn(query_embedding, neg_samples_embedding)
        
        loss = (m + pos_dist.mean(-1) - neg_dist.mean(-1)).clamp(min=0.0)
        return loss


    def homogeneity_loss(self, 
                         query_embedding,
                         same_class_samples_embedding):
        assert query_embedding.shape[1] == 1
        assert same_class_samples_embedding.shape[1] == self.n_same
        
        dist = self.dist_fn(query_embedding, same_class_samples_embedding)
        loss = dist.mean(-1)
        return loss
    
    def load(self, path):
        self.load_state_dict(t.load(path))


class PoincareEmbedBaseline(nn.Module):
    def __init__(self,
                 n_dim=128,
                 n_hidden=32,
                 n_poincare=2,
                 gpu=False,
                 order_loss_mode='classification',
                 **kwargs):

        super(PoincareEmbedBaseline, self).__init__()
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.n_poincare = n_poincare
        self.gpu = gpu
        self.enc = nn.Sequential(
            nn.Linear(self.n_dim, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU()
            )

        self.poincare_embed_norm = nn.Sequential(
            nn.Linear(self.n_hidden, 1),
            nn.Sigmoid())
        
        self.poincare_embed_direction = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_poincare))
        
        self.pred_head = nn.Sequential(
            nn.Linear(self.n_poincare, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, 1)
            )
        
        if self.gpu:
            self.cuda()
        
        if order_loss_mode == 'classification':
            self.order_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif order_loss_mode == 'regression':
            self.order_loss_fn = nn.MSELoss(reduction='none')
            

    def forward(self, inputs):
        _x = self.enc(inputs)
        norm = self.poincare_embed_norm(_x)
        direction = self.poincare_embed_direction(_x)
        direction_normed = direction / t.norm(direction, dim=-1, keepdim=True).clamp(min=eps)
        embedding = direction_normed * norm
        final_norm_pred = self.pred_head(embedding)
        
        return embedding, final_norm_pred


    def train_loss(self, inputs, order_labels=None):
        loss_dict = {}
        embedding, norm = self.forward(inputs)
        
        assert order_labels is not None
        order_loss = self.order_loss_fn(norm.view(*order_labels.shape), order_labels)
        loss_dict['order_loss'] = order_loss.sum()
        loss = order_loss.sum()
        return loss, loss_dict    
        
    def predict(self, dataset):
        if isinstance(dataset, np.ndarray):
            inputs = t.from_numpy(dataset).float()
        else:
            inputs = t.from_numpy(dataset.X).float()
        if self.gpu:
            inputs = inputs.cuda()
        return self.forward(inputs)[1].cpu().data.numpy()
    
    def load(self, path):
        self.load_state_dict(t.load(path))
        

class MLP_pred(nn.Module):
    def __init__(self,
                 n_dim=8985,
                 n_hidden=256,
                 n_poincare=8,
                 gpu=False,
                 order_loss_mode='classification',
                 **kwargs):

        super(MLP_pred, self).__init__()
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.n_poincare = n_poincare
        self.gpu = gpu
        self.enc = nn.Sequential(
            nn.Linear(self.n_dim, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_poincare)
            )
        
        self.pred_head = nn.Sequential(
            nn.Linear(self.n_poincare, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, 1)
            )
            
        if self.gpu:
            self.cuda()

        if order_loss_mode == 'classification':
            self.order_loss_fn = nn.BCEWithLogitsLoss()
        elif order_loss_mode == 'regression':
            self.order_loss_fn = nn.MSELoss()

    def forward(self, inputs):
        z = self.enc(inputs)
        norm_pred = self.pred_head(z)
        
        return z, norm_pred


    def train_loss(self, inputs, order_labels=None):
        loss_dict = {}
        embedding, norm = self.forward(inputs)
        
        assert order_labels is not None
        order_loss = self.order_loss_fn(norm.view(*order_labels.shape), order_labels)
        loss_dict['order_loss'] = order_loss.sum()
        loss = order_loss.sum()
        return loss, loss_dict

        
    def predict(self, dataset):
        if isinstance(dataset, np.ndarray):
            inputs = t.from_numpy(dataset).float()
        else:
            inputs = t.from_numpy(dataset.X).float()
        if self.gpu:
            inputs = inputs.cuda()
        return self.forward(inputs)[1].cpu().data.numpy()


    def load(self, path):
        self.load_state_dict(t.load(path))


def train(model, 
          dataset, 
          weight_save_path=None,
          lr=0.001, 
          batch_size=16, 
          n_epochs=100,
          weight_decay=0.,
          **kwargs):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = t.optim.Adam(model.parameters(), 
                             lr=lr, 
                             betas=(.9, .999), 
                             weight_decay=weight_decay)
    model.train()
    model.zero_grad()
    # min_loss = None
    
    for epoch in range(n_epochs):
        print('start epoch %d' % epoch)
        full_loss_dict = {}
        for batch in loader:
            if isinstance(batch, t.Tensor):
                inputs, order_labels = batch.float(), None
            elif isinstance(batch, list) and isinstance(batch[0], t.Tensor):
                inputs, order_labels = batch[0].float(), batch[1].float()
                
            if model.gpu:
                inputs = inputs.cuda()
                if not order_labels is None:
                    order_labels = order_labels.cuda()
              
            loss, loss_dict = model.train_loss(inputs, order_labels=order_labels)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            
            for k in loss_dict:
                if not k in full_loss_dict:
                    full_loss_dict[k] = 0.
                full_loss_dict[k] += loss_dict[k]
        # if min_loss is None or loss_ave < min_loss:
        #     min_loss = loss_ave
        #     if not weight_save_path is None:
        #         t.save(model.state_dict(), weight_save_path)
        loss_str = ''
        for k in full_loss_dict:
            loss_str += '%s: %.1f\t' % (k, full_loss_dict[k].item())
        print('epoch %d loss: %s' % (epoch, loss_str))
    return model


def evaluate_embedding_with_xgboost(embedding, 
                                    dataset, 
                                    model=None, 
                                    renorm=False, 
                                    return_model=False):
    classes = set(dataset.y)
    n_classes = len(classes)
    class_mapping = dict(zip(list(classes), range(n_classes)))
    y = [class_mapping[_y] for _y in dataset.y]
    
    if isinstance(embedding, t.Tensor):
        embedding = embedding.cpu().data.numpy()
    
    if renorm:
        norm = np.clip(np.linalg.norm(embedding, axis=1, ord=2), 1e-9, 1)
        new_norm = - np.log(1 + 1e-5 - norm)
        embedding = embedding * (new_norm / norm).reshape((-1, 1))
        
    if model is None:
        model = xgboost.XGBClassifier(n_estimators=3*n_classes, max_depth=embedding.shape[1])
        model = model.fit(embedding, y)
    y_pred = model.predict(embedding)
    accuracy = (y_pred == y).sum() / len(y)
    score = v_measure_score(y, y_pred)
    # print("Accuracy: %.3f\tV-measure: %.3f" % (accuracy, score))
    if return_model:
        return (accuracy, score), model
    else:
        return (accuracy, score)
    
    
def evaluate_embedding_order_with_xgboost(embedding, 
                                          dataset, 
                                          model=None, 
                                          renorm=False, 
                                          return_model=False):
    y = dataset.y_order
    
    if isinstance(embedding, t.Tensor):
        embedding = embedding.cpu().data.numpy()
    
    if renorm:
        norm = np.clip(np.linalg.norm(embedding, axis=1, ord=2), 1e-9, 1)
        new_norm = - np.log(1 + 1e-5 - norm)
        embedding = embedding * (new_norm / norm).reshape((-1, 1))
        
    if model is None:
        model = xgboost.XGBRegressor(n_estimators=20, max_depth=embedding.shape[1])
        model = model.fit(embedding, y)
    y_pred = model.predict(embedding)
    
    score = spearmanr(y, y_pred)
    if return_model:
        return [score], model
    else:
        return [score]



def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)

def weighted_cov(x, y, w):
    return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)

def weighted_corr(x, y, w):
    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))

# Spearman correlation weighted more on stages with smaller number of cells
def weighted_spearman_corr(pred, truth):
    ranked_pred = scipy.stats.rankdata(pred)
    ranked_truth = scipy.stats.rankdata(truth)
    num_cells = collections.Counter(truth)
    weights = []
    for phenotype in truth:
        weights.append(1 / num_cells[phenotype])
    return weighted_corr(ranked_pred, ranked_truth, weights)
