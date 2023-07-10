import numpy as np
import faiss
import torch
import  matplotlib.pyplot as plt

valid_models = ['mrl', 'rr']
valid_datasets = ['1K', '4K', 'A' ,'R' ,'O', 'V2'] # ImageNet versions

def load_embeddings(model, dataset, embedding_dim, arch='resnet50', using_gcloud=True):
    if model == 'mrl':
        config = 'mrl1_e0_ff2048'
    elif model == 'rr': # using rigid representation
        config = f'mrl0_e0_ff{embedding_dim}'
    else:
        raise ValueError(f'Model must be in {valid_models}')
    

    if dataset not in valid_datasets:
        raise ValueError(f'Dataset must be in {valid_datasets}')
    if dataset == 'V2': # ImageNetv2 is only a test set; set database to ImageNet-1K
        dataset = '1K'
    

    if using_gcloud:
        root = f'../../../inference_array/{arch}/'
    else: # using local machine
        root = f'../../inference_array/{arch}/'


    db_npy = dataset + '_train_' + config + '-X.npy'
    query_npy = dataset + '_val_' + config + '-X.npy'
    db_label_npy = dataset + '_train_' + config + '-y.npy'
    query_label_npy = dataset + '_val_' + config + '-y.npy'
    
    database = np.load(root + db_npy)
    queryset = np.load(root + query_npy)
    db_labels = np.load(root + db_label_npy)
    query_labels = np.load(root + query_label_npy)

    faiss.normalize_L2(database)
    faiss.normalize_L2(queryset)
    
    xb = np.ascontiguousarray(database[:, :embedding_dim], dtype=np.float32)
    xq = np.ascontiguousarray(queryset[:, :embedding_dim], dtype=np.float32)
    
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)
    
    return database, queryset, db_labels, query_labels, xb, xq

# Find Duplicate distances for low M (M=1 is k-means) on searched faiss index
def get_duplicate_dist(Ind, Dist):
    k = 100
    duplicates = []
    for i in range(50000):
        indices = Ind[i, :][:k] 
        distances = Dist[i, :][:k]
        unique_distances = np.unique(distances, return_counts=1)[1]
        unique_distances = unique_distances[unique_distances != 1] # remove all 1s (i.e. unique distances)
        duplicates = np.append(duplicates, unique_distances)

    hist = plt.hist(duplicates, bins='auto')
    plt.title(model.split("/")[0]+", D="+str(D)+ ", M=" +str(dim))
    plt.xlabel("Number of 100-NN with same neighbor distance values")
    plt.show()
    print(duplicates[duplicates > 2].sum())
    
# Normalize embeddings
def normalize_embeddings(embeddings, dtype):
    if dtype == 'float32':
        print(np.linalg.norm(embeddings))
        faiss.normalize_L2(embeddings)
        print(np.linalg.norm(embeddings))
        return embeddings
    else:
        pass