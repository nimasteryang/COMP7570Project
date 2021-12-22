import re
import uuid
from os import listdir
from os.path import isfile, join
import os
from graph_extractor_example import remove_comment
from graph_extractor_example import AutoExtractGraph
from graph_extractor_example import PatternExtract_RE, PatternExtract_TS
from graph_extractor_example.graph2vec import extract_node_features, elimination_node, embedding_node, elimination_edge, \
    embedding_edge, construct_vec
import torch
from torch_geometric.data import Data
import tempfile
from graph_extractor_example.vec2onehot import vec2onehot

if __name__ == '__main__':
    export_index = 0
    for contract_number in range(1,41):
        folder = f'data/contract/contract{contract_number}'
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for i in files:
            try:
                test_contract_file = f"{folder}/{i}"
                test_comment_output_file = f"data/no_comment_contact/{i}"
                test_contract = remove_comment.remove_comment(test_contract_file, test_comment_output_file)
                re_pattern = PatternExtract_RE.extract_pattern(test_comment_output_file)
                re_label = None
                if len(re_pattern) == 3:
                    if re_pattern[0] == 1:
                        if re_pattern[1] == 1 and re_pattern[2] == 1:
                            re_label = 1
                        else:
                            re_label = 0
                    else:
                        re_label = 0
                else:
                    print("The extracted patterns are error!")
                ts_pattern = PatternExtract_TS.extract_pattern(test_comment_output_file)
                ts_label = None
                if len(ts_pattern) == 3:
                    if ts_pattern[0] == 1:
                        if ts_pattern[1] == 1 and ts_pattern[2] == 1:
                            ts_label = 1
                        else:
                            ts_label = 0
                    else:
                        ts_label = 0
                else:
                    print("The extracted patterns are error!")

                # print(re_label, ts_label, label)
                # sol_file_name = i
                # print(sol_file_name)
                # path_re = f'data/labels_reentry/{i}'
                # if os.path.isfile(path_re):
                #     re_label = True
                # else:
                #     re_label = False

                # path_ts = f'data/labels_timestamp/{i}'
                # if os.path.isfile(path_ts):
                #     ts_label = True
                # else:
                #     ts_label = False

                # if ts_label or re_label:
                #     label = 1
                #     print(f"FIND a VUL: {i}")
                # else:
                #     label = 0
                label = re_label or ts_label
                node_feature, edge_feature = AutoExtractGraph.generate_graph(test_comment_output_file)
                node_feature = sorted(node_feature, key=lambda x: (x[0]))
                edge_feature = sorted(edge_feature, key=lambda x: (x[2], x[3]))
                node_feature, edge_feature = AutoExtractGraph.generate_potential_fallback_node(node_feature, edge_feature)
                # print(node_feature)
                # print(edge_feature)
                temp_dir = tempfile.TemporaryDirectory()
                # print(temp_dir.name)
                AutoExtractGraph.printResult(temp_dir.name, f'{i}', node_feature, edge_feature)

                node = temp_dir.name + '/node_' + f'{i}'
                edge = temp_dir.name + '/edge_' + f'{i}'
                nodeNum, node_list, node_attribute_list = extract_node_features(node)
                node_encode, var_encode, node_embedding, var_embedding = embedding_node(node_attribute_list)
                edge_list, extra_edge_list = elimination_edge(edge)
                edge_encode, edge_embedding = embedding_edge(edge_list)
                node_vec, graph_edge = construct_vec(edge_list, node_embedding, var_embedding, edge_embedding, edge_encode)
                node_vec = sorted(node_vec, key = lambda x: x[0])
                index_map = {}
                edge_index = []
                edge_attr = []
                x = []
                for index, vec in enumerate(node_vec):
                    index_map[vec[0]] = index
                    x.append(vec[1])
                for index, edge in enumerate(edge_encode):
                    edge_index.append([index_map.get(edge[0]), index_map.get(edge[1])])
                    edge_attr.append(edge[2])
                edge_index_tensor = torch.tensor(edge_index,dtype=torch.long)
                pyg_graph = Data(x=torch.tensor(x,dtype=torch.float),edge_index=edge_index_tensor.t().contiguous(),edge_attr=torch.tensor(edge_attr,dtype=torch.long),y=torch.tensor([label],dtype=torch.long))
                torch.save(pyg_graph, f"data/graph_data_expert_label/data_{export_index}.pt")
                export_index += 1
                # use temp_dir, and when done:
                temp_dir.cleanup()
            except:
                print(f"error in {i}")