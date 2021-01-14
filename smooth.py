import os
import numpy as np
import csv


def euclidean_similarity(si, sj):
    return -np.linalg.norm(si - sj, ord=2)


def cosine_similarity(si, sj):
    return np.dot(si, sj) / (np.sqrt(np.dot(si, si)) * np.sqrt(np.dot(sj, sj)))


def find_kNN(samples, pool, k=20, sim_fn=euclidean_similarity):
    assert len(samples.shape) == len(pool.shape) == 2
    assert samples.shape[1] == pool.shape[1]
    assert pool.shape[0] > k

    kNN_inds = []
    for s in samples:
        sims = [sim_fn(s, p) for p in pool]
        kNN_inds.append(np.argsort(sims)[-k:])
    return np.array(kNN_inds)



def smooth_test_with_train(test_Xs, 
                           train_Xs, 
                           model=None,
                           neighbor_mode='feature', # feature or embedding
                           average_mode='feature', # feature or embedding or label
                           k=20,
                           sim_fn=euclidean_similarity)

    test_embedding = None
    train_embedding = None
    if model is not None
        test_inputs = t.from_numpy(test_Xs).float()
        train_inputs = t.from_numpy(train_Xs).float()
        if model.gpu:
            test_inputs = test_inputs.cuda()
            train_inputs = train_inputs.cuda()
        test_embedding = model.forward(test_inputs)[0].cpu().data.numpy()
        train_embedding = model.forward(train_inputs)[0].cpu().data.numpy()
    
    if neighbor_mode == 'feature':
        test_keys = test_Xs
        pool = train_Xs
    elif neighbor_mode == 'embedding':
        test_keys = test_embedding
        pool = train_embedding
        assert test_keys is not None

    kNN_inds = find_kNN(test_keys, pool, k=k, sim_fn=sim_fn)
    if average_mode == 'feature':
        new_test_Xs = [np.stack([train_Xs[i] for i in kNNs], 0).mean(0) for kNNs in kNN_inds]
        return np.stack(new_test_Xs, 0)
    elif average_mode == 'embedding':
        assert model is not None
        new_test_embedding = [np.stack([train_embedding[i] for i in kNNs], 0).mean(0) for kNNs in kNN_inds]
        return np.stack(new_test_embedding, 0)
    elif average_mode == 'label':
        assert model is not None
        train_preds = model.pred_head(train_embedding)
        new_test_preds = [np.mean([train_preds[i] for i in kNNs], 0).mean(0) for kNNs in kNN_inds]
        return np.array(new_test_preds)


