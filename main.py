import os
import time
import numpy as np
from numpy import genfromtxt
from numpy.linalg import eig

import scipy.linalg as la
from sklearn.cluster import SpectralClustering
from sklearn import metrics

import pandas as pd
from pandas import DataFrame as df

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphConv, GlobalAttention, GatedGraphConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.loader import DataLoader


def kercov(dataset,folder_path):
    covariance_matrix_list = [] # creating an empty list
    reg = 0.000001
    for data_file in sorted(os.listdir(folder_path)):
        current_sample = genfromtxt(folder_path+'/'+data_file, delimiter=',') # reading the files in the order of occurence in the folder        
        
        current_sample_new = np.zeros((current_sample.shape[0]-1,current_sample.shape[1]))
        for i in range(current_sample_new.shape[0]):
            current_sample_new[i,:] = current_sample[i,:] - current_sample[i+1,:]
        for kk in range (current_sample_new.shape[1]):
            current_sample_new[:,kk] = current_sample_new[:,kk] / la.norm(current_sample_new[:,kk])
        sigma = np.mean(np.sqrt(abs(current_sample_new)))
        T = current_sample_new.shape[0]
        K = np.exp(np.transpose(current_sample_new)/(sigma**2))
        P = np.copy(K)
        covariance_current_sample = (1/(T-1))*P@(np.identity(T)/T-np.ones(T))@np.transpose(P)
        eigval,eigvec = eig(covariance_current_sample)
        covariance_current_sample = eigvec@np.diag(np.real(np.log((abs(eigval))+reg)))@np.transpose(np.conjugate(eigvec))
        uu = int(np.shape(covariance_current_sample)[1] *(np.shape(covariance_current_sample)[1]+1)/2) # number of lower/ upper triangular elements
        covariance_vector_current_sample = (np.real(covariance_current_sample[np.tril_indices(np.shape(covariance_current_sample)[1])]))
        covariance_vector_current_sample = (covariance_vector_current_sample - min(covariance_vector_current_sample))/(max(covariance_vector_current_sample)-min(covariance_vector_current_sample))
        covariance_matrix_list.append(covariance_vector_current_sample) # saving the upper tirangular values as a vector
    covariance_matrix = np.array(covariance_matrix_list) # converting the list of covaraince matrix to a numpy array
    covariance_matrix = np.transpose(covariance_matrix) # to have each covaraince vector as columns of the matrix
    return(covariance_matrix)



class GCN_GRU(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, out_size):
        super(GCN_GRU, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.gru = torch.nn.GRU(out_channels, out_size, batch_first=True)
        self.attention = AttentionalAggregation(nn.Sequential(nn.Linear(out_size, 1), nn.Tanh()))
        self.classify = nn.Linear(out_size, out_channels)

    def forward(self, data_list):
        x_list = []
        for data in data_list:
            x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
            x = self.conv1(x, edge_index,edge_weight)
            x = x.relu()
            x = self.conv2(x, edge_index,edge_weight)
            x = x.relu()
            x = x.mean(dim=0) # Global pooling over all nodes
            x_list.append(x)
        x = torch.stack(x_list, dim=0) # Stack features from all frames along the time dimension
        x, _ = self.gru(x) # Apply GRU to temporal sequence of feature vectors
        x = self.attention(x)
        x = x[:,:]
        return x


def GNN_feature(dataset,folder_path,num_nodes,edge_index,edge_index_final):
    final_features = []
    for data_file in sorted(os.listdir(folder_path)):
        filename = folder_path+'/'+data_file
        x = pd.read_csv(filename, header = None) # reading the csv file as pandas data frame without header
        x.insert(0,'-3',0)
        x.insert(1,'-2',0)
        x.insert(2,'-1',0)
        x = torch.tensor(x.values, dtype=torch.float) # converting the pandas dataframe to a pytorch tensor

        # reshape the data frame into a tensor with dimensions (num_frames, num_nodes, 3) - each frontal slice contains a frame 
        x = torch.reshape(x, (x.shape[0], -1, 3))
        graph_data = []
        for i in range(x.shape[0]):
            node_feature = x[i]
            edge_feature = []
            for a, b in (edge_index):
                z = torch.norm(node_feature[a] - node_feature[b]) # 1.0
                edge_feature.append(z)
            edge_feature = torch.tensor(edge_feature)
            graph_data_temp = Data(x=node_feature,edge_index=edge_index_final,edge_attr=edge_feature)
            graph_data.append(graph_data_temp)
        model = GCN_GRU(num_features=3, hidden_channels=32, out_channels=20, out_size=32)
        model.load_state_dict(torch.load("G3D_32.pth"))
        features = model(graph_data)
        features=features.detach().numpy()
        features = np.transpose(features)
        final_features.append(features)
    final = np.array(final_features)
    final = np.transpose(final)
    return(final[0])


def main():
    t_i=time.time()

    lambda1 = 0.01
    lambda2 = 15
    n_d = 70
    ksize = 7
    tol = 1e-4
    maxIter = 150
    stepsize = 0.01

    nmi = np.array([])
    ari = np.array([])
    fmi = np.array([])

    print("Starting.....")
    # Reading a particular dataset, corresponding labels, and calculating the covaraince matrix
    dataset = input('Enter the dataset name\n') # deciding the dataset to use
    folder_path =  # set the path
    labels_true = # provide the labels
    if dataset == "F3D":
      num_nodes = 15
      edge_index = [[0,2], [2,0], [2,1], [1,2], [2,3], [3,2], [3,4], [4,3], [4,5], [5,4], [2,6], [6,2], [6,7], [7,6], [7,8], [8,7], [0,9], [9,0], [9,10], [10,9], [10,11], [11,10], [0,12], [12,0], [12,13], [13,12], [13,14], [14,13]]
    elif dataset == "UTK" or "MSRC" or "G3D":
      num_nodes = 20
      edge_index = [[0,1], [1,0], [1,2], [2,1], [2,3], [3,2], [2,4], [4,2], [4,5], [5,4], [5,6], [6,5], [6,7], [7,6], [2,8], [8,2], [8,9], [9,8], [9,10], [10,9], [10,11], [11,10], [0,12], [12,0], [12,13], [13,12], [13,14], [14,13], [14,15], [15,14], [0,16], [16,0], [16,17], [17,16], [17,18], [18,17], [18,19], [19,18]]
    elif dataset == "MSRA":
      num_nodes = 20
      edge_index = [[0,4], [4,0], [4,3], [3,4], [3,19], [19,3], [3,1], [1,3], [1,7], [7,1], [7,9], [9,7], [9,11], [11,9], [3,2], [2,3], [2,8], [8,2], [8,10], [10,8], [10,12], [12,10], [0,5], [5,0], [5,13], [13,5], [13,15], [15,13], [15,17], [17,15], [0,6], [6,0], [6,14], [14,6], [14,16], [16,14], [16,18], [18,16]]
    elif dataset == "HDM14":
      num_nodes = 31
      edge_index = [[0,11], [11,0], [11,12], [12,11], [12,13], [13,12], [13,14], [14,13], [14,15], [15,14], [15,16], [16,15], [13,17], [17,13], [17,18], [18,17], [18,19], [19,18], [19,20], [20,19], [20,23], [23,20], [20,21], [21,20], [21,22], [22,21], [13,24], [24,13], [24,25], [25,24], [25,26], [26,25], [26,27], [27,26], [27,30], [30,27], [27,28], [28,27], [28,29], [29,28], [0,1], [1,0], [1,2], [2,1], [2,3], [3,2], [3,4], [4,3], [4,5], [5,4], [0,6], [6,0], [6,7], [7,6], [7,8], [8,7], [8,9], [9,8], [9,10], [10,9]]
    edge_index_final = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    clusters = len(np.unique(labels_true))
    X1 = GNN_feature(dataset,folder_path,num_nodes,edge_index,edge_index_final)
    X2 = kercov(dataset,folder_path)
    X = np.concatenate((X1, X2), axis=0)

    d,n_x = X.shape
    rho = stepsize
    dsize = n_d
    np.random.seed(1)
    D = np.random.rand(d, dsize).astype(float)
    Z = np.random.rand(dsize, n_x).astype(float)
    V = np.zeros((dsize, n_x)).astype(float)
    Y1 = np.zeros((d, dsize)).astype(float)
    Y2 = np.zeros((dsize, n_x)).astype(float)

    ## --Define the weight matrix-- ##
    k = ksize;
    k2 = (k-1)/2;
    W = np.zeros((n_x, n_x))
    for i in range(n_x):
        for j in range(n_x):
            if abs(i-j) <=k2:
                W[i][j]=1
            else:
                W[i][j]=0
    for i in range(n_x):
        W[i][i] = 0

    alpha = 1e-1
    err = np.zeros((maxIter,1));
    for iteration in range(maxIter):
        ## --Construct Laplacian matrix-- ##
        DD = np.diag(sum(W));
        L = DD - W;

        f_old = la.norm(X-(D@Z), 'fro')

        ## --Update U-- ##
        U = (X@np.transpose(np.conjugate(V)) - Y1 + alpha*D)@np.linalg.inv((V@np.transpose(np.conjugate(V))) + (alpha*np.eye(dsize)))

        ## --V update using Sylvester function-- ##
        VTEMP1 = (np.transpose(np.conjugate(U))@U) + (((2*lambda1)+alpha)*np.eye(dsize))
        VTEMP2 = 2*lambda2*L # OR lambda2*(L+np.transpose(L))
        VTEMP3 = (np.transpose(np.conjugate(U))@X) - Y2 + (alpha*Z)
        V = la.solve_sylvester(VTEMP1,VTEMP2,VTEMP3)

        ## --Update D-- ##
        D = U + (Y1/alpha)
        for kk in range (D.shape[1]):
            D[:,kk] = D[:,kk] / la.norm(D[:,kk])

        ## --Update Z-- ##
        Z = V + (Y2/alpha)
        
        f_new = la.norm(X-(D@Z), 'fro')

        if (abs(f_new - f_old)/max(1,abs(f_old)) < tol) and (np.count_nonzero(D)>0):
            break
        else:
            err[iteration] = abs(f_new - f_old)/max(1,abs(f_old));
            Y1 = Y1 + rho*alpha*(U - D);
            Y2 = Y2 + rho*alpha*(V - Z);
            
        vecNorm = sum(pow(Z,2))
        vecNorm = vecNorm.reshape(1,vecNorm.shape[0])
        Q = (np.transpose(np.conjugate(Z))@Z) / (np.transpose(np.conjugate(vecNorm))@vecNorm + 1e-6);


        clustering = SpectralClustering(n_clusters=clusters,affinity='precomputed',assign_labels='discretize',random_state=0).fit(Q)
        labels_pred = clustering.labels_
        nmi = np.append(nmi,metrics.adjusted_mutual_info_score(labels_true,labels_pred))
        ari = np.append(ari,metrics.adjusted_rand_score(labels_true,labels_pred))
        fmi = np.append(fmi,metrics.fowlkes_mallows_score(labels_true, labels_pred))
        No_of_iterations = iteration;
    t_o=time.time()
    time_final = t_o - t_i

        
    nmi_mean = np.mean(nmi[:])
    ari_mean = np.mean(ari[:])
    fmi_mean = np.mean(fmi[:])
    nmi_std = np.std(nmi[:])
    ari_std = np.std(ari[:])
    fmi_std = np.std(fmi[:])
    nmi_final = nmi[len(nmi)-1]
    ari_final = ari[len(ari)-1]
    fmi_final = fmi[len(fmi)-1]

    print('Mean NMI=%f'%nmi_mean)
    print('Mean ARI=%f'%ari_mean)
    print('Mean FMI=%f'%fmi_mean)
    print('std NMI=%f'%nmi_std)
    print('std ARI=%f'%ari_std)
    print('std FMI=%f'%fmi_std)
    print('final NMI=%f'%nmi_final)
    print('final ARI=%f'%ari_final)
    print('final FMI=%f'%fmi_final)
    print('No of iterations=%f'%No_of_iterations)
    print('time taken=%f'%time_final)

if __name__ == "__main__":
    main()
