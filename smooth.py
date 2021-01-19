import os
import numpy as np
import csv
import torch

def euclidean_similarity(si, sj):
    return -np.linalg.norm(si - sj, ord=2, axis=2)


def cosine_similarity(si, sj):
    return (si * sj).sum(2) / (np.sqrt((si**2).sum(2)) * np.sqrt((sj**2).sum(2)))


def calculate_block_similarity(si, sj, sim_fn=euclidean_similarity):

    assert si.shape[1] == 1
    assert sj.shape[0] == 1
    n_dim = si.shape[-1]
    
    batch_size = int(np.floor(1e6 / n_dim))
    if batch_size > sj.shape[1]:
        j_batch_size = sj.shape[1]
        i_batch_size = batch_size // j_batch_size
    else:
        j_batch_size = batch_size
        i_batch_size = 1

    n_i_batches = int(np.ceil(si.shape[0] / i_batch_size))
    n_j_batches = int(np.ceil(sj.shape[1] / j_batch_size))

    total_combined = []
    for i in range(n_i_batches):
        row_combined = []
        for j in range(n_j_batches):
            _si = si[(i*i_batch_size):((i+1)*i_batch_size)]
            _sj = sj[:, (j*j_batch_size):((j+1)*j_batch_size)]
            block_sim = sim_fn(_si, _sj)
            row_combined.append(block_sim)
        row_combined = np.concatenate(row_combined, 1)
        total_combined.append(row_combined)
    total_combined = np.concatenate(total_combined, 0)
    return total_combined


def find_kNN(samples, pool, k=20, sim_fn=euclidean_similarity):
    assert len(samples.shape) == len(pool.shape) == 2
    assert samples.shape[1] == pool.shape[1]
    assert pool.shape[0] > k
    n_dim = samples.shape[1]
    
    sim_mat = calculate_block_similarity(samples.reshape((-1, 1, n_dim)), 
                                         pool.reshape((1, -1, n_dim)), 
                                         sim_fn=sim_fn)
    kNN_inds = [np.argsort(row_sim)[-k:] for row_sim in sim_mat]
    return np.array(kNN_inds)


def get_embeddings(feats, model):
    feats = torch.from_numpy(feats.astype(float)).float()
    if model.gpu:
        feats = feats.cuda()
    embeddings, preds = model.forward(feats)
    embeddings = embeddings.cpu().data.numpy()
    return embeddings


def get_preds(feats, model):
    feats = torch.from_numpy(feats.astype(float)).float()
    if model.gpu:
        feats = feats.cuda()
    embeddings, preds = model.forward(feats)
    preds = preds.cpu().data.numpy()
    return preds


def get_preds_from_embeddings(embeddings, model):
    embeddings = torch.from_numpy(embeddings.astype(float)).float()
    if model.gpu:
        embeddings = embeddings.cuda()
    preds = model.pred_head(embeddings)
    preds = preds.cpu().data.numpy()
    return preds


def smoothing_by_average_kNN(pool, kNN_inds):
    assembled_samples = [np.stack([pool[i] for i in kNNs], 0).mean(0) for kNNs in kNN_inds]
    return np.stack(assembled_samples, 0)



