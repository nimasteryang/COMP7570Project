import os
import torch
import numpy as np

"""
Here is the method for extracting security patterns of timestamp dependence.
"""

patterns = {"pattern1": 1, "pattern2": 2, "pattern3": 3}

patterns_flag = {"100", "010", "001"}


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
            if text.split()[0] == "function" or text.split()[0] == "function()":
                function_list.append([text])
                flag += 1
            elif len(function_list) > 0 and ("function" in function_list[flag][0]):
                function_list[flag].append(text)

    return function_list


# Position the call.value to generate the graph
def extract_pattern(filepath):
    allFunctionList = split_function(filepath)  # Store all functions
    timeStampList = []  # Store all W functions that call call.value
    otherFunctionList = []  # Store functions other than W functions
    pattern_list = []

    # Store other functions without W functions (with block.timestamp)
    for i in range(len(allFunctionList)):
        flag = 0
        for j in range(len(allFunctionList[i])):
            text = allFunctionList[i][j]
            if 'block.timestamp' in text:
                timeStampList.append(allFunctionList[i])
                flag += 1
        if flag == 0:
            otherFunctionList.append(allFunctionList[i])

    ################   pattern 1: timestampInvocation  #######################
    if len(timeStampList) != 0:
        pattern_list.append(1)
    else:
        pattern_list.append(0)
        pattern_list.append(0)
        pattern_list.append(0)

    ################   pattern 2: timestampAssign      #######################
    for i in range(len(timeStampList)):
        TimestampFlag1 = 0
        VarTimestamp = None

        if len(pattern_list) > 1:
            break

        for j in range(len(timeStampList[i])):
            text = timeStampList[i][j]
            if 'block.timestamp' in text:
                TimestampFlag1 += 1
                VarTimestamp = text.split("=")[0]
            elif TimestampFlag1 != 0:
                if VarTimestamp != " " or "":
                    if VarTimestamp in text:
                        pattern_list.append(1)
                        break
                    elif j + 1 == len(timeStampList[i]) and len(pattern_list) == 1:
                        pattern_list.append(0)
                else:
                    pattern_list.append(0)
                    break

    ################  pattern 3: timestampContamination  #######################
    for i in range(len(timeStampList)):
        TimestampFlag2 = 0
        VarTimestamp = None

        if len(pattern_list) > 2:
            break

        for j in range(len(timeStampList[i])):
            text = timeStampList[i][j]
            if 'block.timestamp' in text:
                VarTimestamp = text.split("=")[0]
                TimestampFlag2 += 1
                if 'return' in text:
                    pattern_list.append(1)
                    break
            elif TimestampFlag2 != 0:
                if VarTimestamp in text and 'return' in text:
                    pattern_list.append(1)
                    break
                elif j + 1 == len(timeStampList[i]) and len(pattern_list) == 2:
                    pattern_list.append(0)

    return pattern_list



