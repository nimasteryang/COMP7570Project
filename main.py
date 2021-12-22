
import bz2
import os
from tensorboardX import SummaryWriter 
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    average_precision_score, classification_report, roc_curve, auc, top_k_accuracy_score, ndcg_score
from torch import nn
import sys
import gc
import _pickle as cPickle
import models
from tqdm import tqdm
from torch_geometric.data import Dataset, DataLoader, InMemoryDataset
import argparse
import logging
from torch_geometric.nn import GCN, GIN, GAT
import copy
from torch_geometric.nn import global_max_pool, global_mean_pool
import seaborn as sns
from sklearn.manifold import TSNE

from matplotlib import cm
def draw_tsne(_output,_target):
    plt.figure(figsize=(10, 8))
    np_output = np.stack(_output,axis=1)[0]
    # print(np_output.shape)
    # print(_target)
    tsne = TSNE(n_components=2,random_state=42)
    tsne_proj = tsne.fit_transform(np_output)
    plt.cla()
    X, Y = tsne_proj[:, 0], tsne_proj[:, 1]
    for x, y, s in zip(X, Y, _target):
        if s == 0:
            plt.plot(x,y,'o',color = 'b',label='Non-vulnerable',alpha=0.5)
        else:
            plt.plot(x,y,'x',color = 'r',label='Vulnerable',alpha=0.5)
    plt.show()

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, percentage=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.graph_files = [f for f in os.listdir(root) if
                            os.path.isfile(os.path.join(root, f))]
        if percentage != None:
            origin_len = len(self.graph_files)
            self.graph_files = self.graph_files[:int(percentage * origin_len)]
            print(f'{percentage} dataset from {origin_len} to {len(self.graph_files)}')
    @property
    def processed_file_names(self):
        return self.graph_files

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.root, 'data_{}.pt'.format(idx)))
        return data

def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    # return MAE
    return np.mean(abs(v1 - v2), axis=1)

def draw_tsne(_output,_target):
    np_output = np.stack(_output)
    print(np_output.shape)
    print(_target)


from matplotlib import cm
def draw_tsne(_output,_target):
    plt.figure(figsize=(10, 8))
    np_output = np.stack(_output,axis=1)[0]
    # print(np_output.shape)
    # print(_target)
    tsne = TSNE(n_components=2,random_state=42)
    tsne_proj = tsne.fit_transform(np_output)
    plt.cla()
    X, Y = tsne_proj[:, 0], tsne_proj[:, 1]
    for x, y, s in zip(X, Y, _target):
        if s == 0:
            plt.plot(x,y,'o',color = 'b',label='Non-vulnerable',alpha=0.5)
        else:
            plt.plot(x,y,'x',color = 'r',label='Vulnerable',alpha=0.5)
    plt.show()

def main(processed_dir,percentage=None):
    # for rubust experiment,can test with different data split
    train_dataset = MyOwnDataset(processed_dir + 'train',percentage=percentage)
    valid_dataset = MyOwnDataset(processed_dir + 'valid')
    test_dataset = MyOwnDataset(processed_dir + 'test')

    train_loader = DataLoader(train_dataset, batch_size=1, batch_sampler=None,
                            shuffle=True) 
    valid_loader = DataLoader(valid_dataset, batch_size=1, batch_sampler=None, shuffle=True)                          
    test_loader = DataLoader(test_dataset, batch_size=1, batch_sampler=None, shuffle=False)
    print("dir",processed_dir," train", len(train_dataset), "test", len(test_dataset))
    seed = 2
    print(f"seed {seed}")
    torch.manual_seed(seed)
    device = 'cpu'
    # graph_model = GIN(in_channels=250, hidden_channels=250, num_layers=1)
    graph_model = GCN(in_channels=250, hidden_channels=250, num_layers=1)
    ae_model = models.AE(n_layers=3, first_layer_size=250)
    graph_model.to(device)
    ae_model.to(device)
    print(graph_model)
    print(ae_model)
    print(device)
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-6)
    criterion = nn.MSELoss(reduction='mean')
    train(train_loader,graph_model,ae_model,optimizer,criterion,device)
    threshold = fine_tune(valid_loader,graph_model,ae_model,optimizer,criterion,device)
    testing_f1 = testing(test_loader,graph_model,ae_model,criterion,threshold,device)
    return testing_f1
    


def train(train_loader,graph_model,ae_model,optimizer,criterion,device):
    # writer = SummaryWriter()
    F1_avg = 0
    train_loss = 0
    predictions = []
    actual = []
    graph_model.eval()
    ae_model.train()
    # for e in range(4):
    for index, graph in enumerate(tqdm(train_loader)):
        if device != 'cpu':
            graph = graph.to(device)
        # if index % 10000 == 0:
        #     print(f"{index} step curr_loss:{train_loss/(index+1)}")
        #     train_loss = 0
        if index % 100 == 0 and index != 0:
            # writer.add_scalar("loss", train_loss/(100), global_step=index, walltime=None)
            train_loss = 0
        graph_feature = graph_model(graph.x, graph.edge_index)
        graph_feature = global_max_pool(graph_feature,
                                        torch.zeros(graph_feature.shape[0], dtype=int, device=graph_feature.device))
        actual.append(graph_feature.cpu().detach().numpy())
        optimizer.zero_grad()
        outputs = ae_model(graph_feature)
        predictions.append(outputs.cpu().detach().numpy())
        loss = criterion(outputs, graph_feature)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

def fine_tune(valid_loader,graph_model,ae_model,optimizer,criterion,device):
    #fine tune
    print("Fine tune >")
    ae_model.eval()
    graph_model.eval()
    loss_dist = []
    loss_sc_vul = []
    loss_sc_nonvul = []
    test_predictions = []
    test_actual = []
    test_targets = []
    test_outputs = []
    output_store = []
    for index, graph in enumerate(tqdm(valid_loader)):
        if device != 'cpu':
            graph = graph.to(device)
        graph_feature = graph_model(graph.x, graph.edge_index)
        graph_feature = global_max_pool(graph_feature,
                                        torch.zeros(graph_feature.shape[0], dtype=int, device=graph_feature.device))
        optimizer.zero_grad()
        output = ae_model(graph_feature)
        test_outputs.append(output.cpu().detach().numpy())
        test_actual.append(graph_feature.cpu().detach().numpy())
        loss = criterion(output, graph_feature)
        loss_dist.append(loss.item())
        target = graph.y.cpu().detach().numpy()[0]
        test_targets.append(target)
        if target == 0:
            loss_sc_nonvul.append(loss.item())
        else:
            loss_sc_vul.append(loss.item())

    threshold=np.amin(loss_dist)
    upper_threshold = np.amax(loss_dist)
    best_threshold = 0
    f1=0
    best_f1 = 0
    best_confusion = None
    recall=0
    accuracy=0
    iterations = 1000
    for i in tqdm(range(0,iterations)):
        if iterations == 0:
            break
        # print ('**************************')
        # print (threshold)
        threshold+=(upper_threshold-threshold)/1000
        # print (threshold)
        y_pred = [1 if e > threshold else 0 for e in loss_dist]
        conf_matrix = confusion_matrix(test_targets, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = 1.*tp/(tp+fp)
        recall = 1.*tp/(tp+fn)
        f1=(2*recall*precision)/(recall+precision)
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1
            best_confusion = conf_matrix
    print("valid set result:")
    print(best_f1)
    print(best_confusion)
    print(best_threshold)
    print('--------------------------------')
    return best_threshold


def testing(test_loader,graph_model,ae_model,criterion,best_threshold,device):
    print("test set >")
    ae_model.eval()
    graph_model.eval()
    loss_sc_nonvul = []
    loss_sc_vul = []
    test_predictions = []
    test_targets = []
    test_outputs = []
    for index, graph in enumerate(tqdm(test_loader)):
        if device != 'cpu':
            graph = graph.to(device)
        graph_feature = graph_model(graph.x, graph.edge_index)
        graph_feature = global_max_pool(graph_feature,
                                        torch.zeros(graph_feature.shape[0], dtype=int, device=graph_feature.device))
        output = ae_model(graph_feature)
        test_outputs.append(output.cpu().detach().numpy())
        # test_actual.append(graph_feature.cpu().detach().numpy())
        loss = criterion(output, graph_feature)
        # loss_dist.append(loss.item())
        target = graph.y.cpu().detach().numpy()[0]
        test_targets.append(target)
        if target == 0:
            loss_sc_nonvul.append(loss.item())
        else:
            loss_sc_vul.append(loss.item())
        if loss < best_threshold:
            test_predictions.append(0)
        else:
            test_predictions.append(1)


    conf_matrix = confusion_matrix(test_targets, test_predictions)
    acc = accuracy_score(test_targets, test_predictions) 
    precision = precision_score(test_targets, test_predictions)
    recall = recall_score(test_targets, test_predictions)
    print("acc:",acc)
    print("recall",recall)
    print("precision",precision)
    print(classification_report(test_targets, test_predictions))
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = 1.*tp/(tp+fp)
    recall = 1.*tp/(tp+fn)
    f1=(2*recall*precision)/(recall+precision)
    print(f1)
    return f1



if __name__ == "__main__" :
    # writer = SummaryWriter()
    # for p in [0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.8]:
    #     f1 = main(processed_dir = 'data/processed_data_static/',percentage=p)
    #     writer.add_scalar("f1", f1, p*100, walltime=None)
    main(processed_dir = 'data/processed_data_static/')