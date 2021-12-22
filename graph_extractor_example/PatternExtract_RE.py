import re
import os
import torch
import numpy as np


"""
Extract expert patterns of smart contract reentrancy vulnerability.
"""


# split all functions of contracts
def split_function(filepath):
    function_list = []
    f = open(filepath, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    flag = -1

    for line in lines:
        text = line.strip()
        if len(text) > 0 and text != "\n":
            if text.split()[0] == "function" or text.split()[0] == "constructor":
                function_list.append([text])
                flag += 1
            elif len(function_list) > 0 and ("function" or "constructor" in function_list[flag][0]):
                function_list[flag].append(text)

    return function_list


# Position the call.value to generate the graph
def extract_pattern(filepath):
    allFunctionList = split_function(filepath)  # Store all functions
    callValueList = []  # Store all functions that call call.value
    otherFunctionList = []  # Store functions other than the functions that contains call.value
    pattern_list = []

    # Store functions other than W functions (with .call.value)
    for i in range(len(allFunctionList)):
        flag = 0
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if '.call.value' in text:
                callValueList.append(allFunctionList[i])
                flag += 1
        if flag == 0:
            otherFunctionList.append(allFunctionList[i])

    ################  pattern 1: callValueInvocation  #######################
    if len(callValueList) != 0:
        pattern_list.append(1)
    else:
        pattern_list.append(0)
        pattern_list.append(0)
        pattern_list.append(0)

    ################   pattern 2: balanceDeduction   #######################
    for i in range(len(callValueList)):
        CallValueFlag1 = 0
        if len(pattern_list) > 1:
            break

        for j in range(len(callValueList[i])):
            text = callValueList[i][j]
            if '.call.value' in text:
                CallValueFlag1 += 1
            elif CallValueFlag1 != 0:
                text = text.replace(" ", "")
                if "-" in text or "-=" in text or "=0" in text:
                    pattern_list.append(1)
                    break
                elif j + 1 == len(callValueList[i]) and len(pattern_list) == 1:
                    pattern_list.append(0)

    ################   pattern 3: enoughBalance     #######################
    for i in range(len(callValueList)):
        CallValueFlag2 = 0
        param = None
        if len(pattern_list) > 2:
            break

        for j in range(len(callValueList[i])):
            text = callValueList[i][j]
            if '.call.value' in text:
                CallValueFlag2 += 1
                param = re.findall(r".call.value\((.+?)\)", text)[0]
            elif CallValueFlag2 != 0:
                if param in text:
                    pattern_list.append(1)
                    break
                elif j + 1 == len(callValueList[i]) and len(pattern_list) == 2:
                    pattern_list.append(0)

    return pattern_list


# def extract_feature_by_fnn(outputPathFNN, pattern1, pattern2, pattern3):
#     pattern1 = torch.Tensor(pattern1)
#     pattern2 = torch.Tensor(pattern2)
#     pattern3 = torch.Tensor(pattern3)
#     model = FFNNP(4, 100, 250)
#
#     pattern1FC = model(pattern1).detach().numpy().tolist()
#     pattern2FC = model(pattern2).detach().numpy().tolist()
#     pattern3FC = model(pattern3).detach().numpy().tolist()
#     pattern_final = np.array([pattern1FC, pattern2FC, pattern3FC])
#
#     np.savetxt(outputPathFNN, pattern_final, fmt="%.6f")
#     np.savetxt(outputPathFNN, pattern_final, fmt="%.6f")