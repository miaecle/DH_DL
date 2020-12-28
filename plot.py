# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:57:33 2020

@author: Zhenqin Wu
"""
import matplotlib
import matplotlib.pyplot as plt
import torch as t
import numpy as np
import collections
import scipy
from labels import ABBR_TO_ORDER
import pandas as pd
import seaborn as sns
from data import HierarchyDataLoader

def plot_embedding(embedding, 
                   dataset, 
                   with_unit_circle=True, 
                   full_view=False,
                   legend=True,
                   renorm=False,
                   subtree=(),
                   size=(6.5, 5)):
    if isinstance(embedding, t.Tensor):
        embedding = embedding.cpu().data.numpy()
    
    if renorm:
        norm = np.clip(np.linalg.norm(embedding, axis=1, ord=2), 1e-9, 1)
        new_norm = - np.log(1 + 1e-5 - norm)
        embedding = embedding * (new_norm / norm).reshape((-1, 1))
        
    plt.clf()
    plt.figure(figsize=size)
    if with_unit_circle:
        circle = plt.Circle((0,0), 1, edgecolor=(0.1, 0.1, 0.1, 0.5), facecolor='none')
        plt.gca().add_artist(circle)
    classes = set(list(dataset.y))
    
    cmap = matplotlib.cm.get_cmap('tab20')
    
    color_maps = []
    points = []
    colors = []
    for c in sorted(classes):
        inds = np.array([i for i, _y in enumerate(dataset.y) if _y == c])
        label = c[-1] if c[:len(subtree)] == subtree else None
        color = hash(str(c)) % 20
        while color in color_maps:
            color = (color + 1)%20
            if len(color_maps) >= 20:
                break
        color_maps.append(color)
        points.append(embedding[inds])
        colors.extend([cmap(color)] * len(inds))
        plt.scatter([0], [0], s=1, color=cmap(color), label=label)
    plt.scatter([0], [0], s=1.1, color='w') # Cover legend marks
    
    points = np.concatenate(points, 0)
    shuffle_inds = np.arange(len(points))
    np.random.shuffle(shuffle_inds)
    points = points[shuffle_inds]
    colors = [colors[i] for i in shuffle_inds]
    
    plt.scatter(points[:, 0], points[:, 1], s=1, c=colors)
    
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    if full_view:
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
    else:
        inds = np.array([i for i, _y in enumerate(dataset.y) if _y[:len(subtree)] == subtree])
        if renorm:
            x_lower = np.percentile(embedding[inds][:, 0], 1) 
            x_upper = np.percentile(embedding[inds][:, 0], 99)
            y_lower = np.percentile(embedding[inds][:, 1], 1)
            y_upper = np.percentile(embedding[inds][:, 1], 99)
            x_interval = x_upper - x_lower
            y_interval = y_upper - y_lower
            plt.xlim(x_lower - x_interval * 0.2, x_upper + x_interval * 0.2)
            plt.ylim(y_lower - y_interval * 0.2, y_upper + y_interval * 0.2)
            
        else:
            x_lower = np.percentile(embedding[inds][:, 0], 20) 
            x_upper = np.percentile(embedding[inds][:, 0], 80)
            y_lower = np.percentile(embedding[inds][:, 1], 20)
            y_upper = np.percentile(embedding[inds][:, 1], 80)
            x_interval = x_upper - x_lower
            y_interval = y_upper - y_lower
            plt.xlim(x_lower - x_interval, x_upper + x_interval)
            plt.ylim(y_lower - y_interval, y_upper + y_interval)
    plt.tight_layout()



def plot_embedding_norm(embedding, dataset, renorm=True):
    if isinstance(embedding, t.Tensor):
        embedding = embedding.cpu().data.numpy()
    
    if len(embedding.shape) == 1:
        norm = embedding
    else:
        norm = np.clip(np.linalg.norm(embedding, axis=1, ord=2), 1e-9, 1)
    
    if renorm:
        norm = - np.log(1 + 1e-5 - norm)
    
    if isinstance(dataset, HierarchyDataLoader):
        phenotypes = [y[-1] for y in dataset.y]
        orders = [ABBR_TO_ORDER[y] if y in ABBR_TO_ORDER else -1. for y in dataset.y]
        phenotypes = list(zip(phenotypes, orders))
        phenotypes_order = sorted(set(phenotypes), key=lambda x: (x[0][:-1], x[1]))
    else:
        phenotypes = dataset
        phenotypes_order = sorted(set(phenotypes))
    
    df = pd.DataFrame({'norm': norm, 'phenotype': phenotypes})
    
    plt.clf()
    f = plt.figure(figsize=(5, 3))
    sns.violinplot(x="phenotype", 
                   y="norm", 
                   data=df, 
                   order=phenotypes_order, 
                   scale='width',
                   width=0.5,
                   linewidth=0.1)
    plt.xticks(rotation=45)
    plt.tight_layout()