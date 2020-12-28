# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:48:41 2020

@author: Zhenqin Wu
"""
import os
import pandas as pd
import numpy as np

###
datasets = os.listdir('data/dropletOntogenyDatasets')
ontogeny_df = pd.read_csv("data/dropletOntogenyDatasets/dropletOntogenyTable.txt", sep='\t')
cell_ids = list(ontogeny_df['RDS_UniqueID'])

for dataset in datasets:
    if not dataset.endswith('_downsampled.tsv'):
        continue
    X_df = pd.read_csv(os.path.join("data/dropletOntogenyDatasets", dataset), sep='\t', index_col=0).T
    
    index = []
    phenotypes = []
    ranks = []
    ontogenetic_order = []
    combined_order = []
    ct = 0
    for sample_id in X_df.index:
        if sample_id not in cell_ids:
            # print("Ontogenetic order is not known for " + dataset, sample_id)
            ct += 1
            continue

        row = ontogeny_df[ontogeny_df.RDS_UniqueID.eq(sample_id)]
        assert len(row) == 1
        index.append(sample_id)
        phenotypes.append(row['Phenotype'].item())
        ranks.append(int(row['Dataset_level_order']))
        ontogenetic_order.append(int(row['Ontogenetic_order']))
        try:
            combined_order.append(int(row['CombinedMouseAndHumanOrder']))
        except ValueError:
            combined_order.append(np.nan)

    print("%d/%d" % (ct, X_df.shape[0]))
    dataset_name = dataset.split('_downsampled')[0]
    y_df = pd.DataFrame({'phenotype': phenotypes, 
                         'rank': ranks, 
                         'ontogenetic_order': ontogenetic_order,
                         'combined_order': combined_order},
                        index=index).to_csv('data/dropletOntogenyDatasets/%s_phenotype.csv' % dataset_name)

    X_df.loc[index].T.to_csv('data/dropletOntogenyDatasets/%s_downsampled_with_ontogeny.tsv' % dataset_name, sep='\t')

###
datasets = os.listdir('data/plateOntogenyDatasets')
ontogeny_df = pd.read_csv("data/plateOntogenyDatasets/plateOntogenyTable.txt", sep='\t')
cell_ids = list(ontogeny_df['Unique_ID'])

for dataset in datasets:
    if not dataset.endswith('_downsampled.tsv'):
        continue
    print(dataset)
    X_df = pd.read_csv(os.path.join("data/plateOntogenyDatasets", dataset), sep='\t', index_col=0).T
    
    index = []
    phenotypes = []
    ranks = []
    ontogenetic_order = []
    combined_order = []
    ct = 0
    for sample_id in X_df.index:
        if dataset.startswith('Kyle'):
            unique_id = sample_id.split('.')[0]
        else:
            unique_id = sample_id.split('_')[0]
        if unique_id not in cell_ids:
            ct += 1
            continue
        row = ontogeny_df[ontogeny_df.Unique_ID.eq(unique_id)]
        assert len(row) == 1
        index.append(sample_id)
        phenotypes.append(row['Phenotype'].item())
        ranks.append(int(row['Dataset_level_order']))
        ontogenetic_order.append(int(row['Ontogenetic_order']))
        try:
            combined_order.append(int(row['CombinedMouseAndHumanOrder']))
        except ValueError:
            combined_order.append(np.nan)
    print("%d/%d" % (ct, X_df.shape[0]))

    dataset_name = dataset.split('_downsampled')[0]
    y_df = pd.DataFrame({'phenotype': phenotypes, 
                         'rank': ranks, 
                         'ontogenetic_order': ontogenetic_order,
                         'combined_order': combined_order},
                        index=index).to_csv('data/plateOntogenyDatasets/%s_phenotype.csv' % dataset_name)

    X_df.loc[index].T.to_csv('data/plateOntogenyDatasets/%s_downsampled_with_ontogeny.tsv' % dataset_name, sep='\t')