
import os
import random

import torch
from tqdm import tqdm

graph_path = f'{os.getcwd()}/data/graph_data_expert_label'
graph_files = [f for f in os.listdir(graph_path) if
               os.path.isfile(os.path.join(graph_path, f))]

index = 0
vul_list = []
nonvul_list = []
for graph_file in tqdm(graph_files):
    graph = torch.load(f'{graph_path}/{graph_file}')
    if graph.y == 0:
        nonvul_list.append(graph_file)
    else:
        vul_list.append(graph_file)
print("vul",len(vul_list))
print("nonvul",len(nonvul_list))

random.shuffle(vul_list)
random.shuffle(nonvul_list)
# print(len(vul_list))
# print(len(nonvul_list))


len_train = len(nonvul_list)-len(vul_list)
len_valid = int(0.2 * len(vul_list))
len_test = int(0.8 * len(vul_list))
train_list = nonvul_list[:len_train]
print("train:",len(train_list))
# remaining_list = nonvul_list[len(nonvul_list)-len(vul_list):] + vul_list

# random.shuffle(remaining_list)
valid_list = nonvul_list[len_train:len_train+len_valid] + vul_list[:len_valid]
test_list = nonvul_list[len_train+len_valid:] + vul_list[len_valid:]
print("valid:",len(valid_list))
print("test:",len(test_list))

for index, train in enumerate(tqdm(train_list)):
    train_graph = torch.load(f'{graph_path}/{train}')
    torch.save(train_graph, f'{os.getcwd()}/data/processed_data_expert/train/data_{index}.pt')

for index, valid in enumerate(tqdm(valid_list)):
    valid_graph = torch.load(f'{graph_path}/{valid}')
    torch.save(valid_graph, f'{os.getcwd()}/data/processed_data_expert/valid/data_{index}.pt')

for index, test in enumerate(tqdm(test_list)):
    test_graph = torch.load(f'{graph_path}/{test}')
    torch.save(test_graph, f'{os.getcwd()}/data/processed_data_expert/test/data_{index}.pt')