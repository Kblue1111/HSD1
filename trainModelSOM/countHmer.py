import random

import sklearn
import json
import os
import pickle
import subprocess
import logging
import numpy as np
import datetime
import pickle
import operator
import math
import sys
import threading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import multiprocessing
from sklearn.model_selection import GridSearchCV
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.neural_network import MLPClassifier
import socket
from collections import Counter

warnings.filterwarnings("ignore")

with open('./config.json', 'r') as configFile:
    configData = json.load(configFile)

faultlocalizationResultPath = configData['faultlocalizationResultPath']
djfeaturePath = configData['djfeaturePath']
djSrcPath = configData['djSrcPath']
outputCleanPath = configData['outputCleanPath']
dynamicFeaturePath = configData['dynamicFeaturePath']
tpydataPath = configData['tpydataPath']
allTestsPath = configData['allTestsPath']


# region mbfl计算公式

def dstar(Akf, Anf, Akp, Anp):
    if (Akp + Anf) == 0:
        return sys.float_info.max
    return math.pow(Akf, 3) / (Akp + Anf)


def dstar_sub_one(Akf, Anf, Akp, Anp):
    if (Akp + Anf) == 0:
        return sys.float_info.max
    return 1 / (Akp + Anf)


def ochiai(Akf, Anf, Akp, Anp):
    if (Akf + Anf) * (Akf + Akp) == 0:
        return 0
    return Akf / math.sqrt((Akf + Anf) * (Akf + Akp))


def ochiai_sub_one(Akf, Anf, Akp, Anp):
    if (Akf + Anf) == 0:
        return 0
    return 1 / math.sqrt(Anf + Akf)


def ochiai_sub_two(Akf, Anf, Akp, Anp):
    if (Akf + Akp) == 0:
        return 0
    return 1 / math.sqrt(Akf+Akp)


def ochiai_sub_three(Akf, Anf, Akp, Anp):
    if (Akf + Anf) * (Akf + Akp) == 0:
        return 0
    return 1 / math.sqrt((Akf + Anf) * (Akf + Akp))


def ochiai_sub_four(Akf, Anf, Akp, Anp):
    if (Akf + Anf) == 0:
        return 0
    return Akf / math.sqrt(Anf + Akf)


def ochiai_sub_five(Akf, Anf, Akp, Anp):
    if (Akf + Akp) == 0:
        return 0
    return Akf / math.sqrt(Akf + Akp)


def gp13(Akf, Anf, Akp, Anp):
    if (2 * Akp + Akf) == 0:
        return 0
    return Akf + (Akf / (2 * Akp + Akf))


def gp13_sub_one(Akf, Anf, Akp, Anp):
    if (2 * Akp + Akf) == 0:
        return 0
    return 1 / (2 * Akp + Akf)


def gp13_sub_two(Akf, Anf, Akp, Anp):
    if (2 * Akp + Akf) == 0:
        return 0
    return Akf / (2 * Akp + Akf)


def op2(Akf, Anf, Akp, Anp):
    # XSQ
    if ((Akp + Anp) + 1) == 0:
        return 0
    return Akf - (Akp / ((Akp + Anp) + 1))


def op2_sub_one(Akf, Anf, Akp, Anp):
    return Akp / ((Akp + Anf) + 1)


def op2_sub_two(Akf, Anf, Akp, Anp):
    return 1 / ((Akp + Anf) + 1)


def jaccard(Akf, Anf, Akp, Anp):
    if (Akf + Anf + Akp) == 0:
        return 0
    return Akf / (Anf + Akf + Akp)


def jaccard_sub_one(Akf, Anf, Akp, Anp):
    if (Akf + Anf + Akp) == 0:
        return 0
    return 1 / (Anf + Akf + Akp)


def russell(Akf, Anf, Akp, Anp):
    # XSQ
    if  (Akp + Anp + Akf + Anf) == 0:
        return 0
    return Akf / (Akp + Anp + Akf + Anf)


def russell_sub_one(Akf, Anf, Akp, Anp):
    return 1 / (Akp + Anp + Akf + Anf)


def turantula(Akf, Anf, Akp, Anp):
    if (Akf + Akp) == 0 or (Akf + Anf) == 0:
        return 0
    if ((Akf + Anf) != 0) and ((Akp + Anp) == 0):
        return 1
    return (Akf / (Akf + Anf)) / ((Akf / (Akf + Anf)) + (Akp / (Akp + Anp)))


def turantula_sub_one(Akf, Anf, Akp, Anp):
    if (Akf + Akp) == 0 or (Akf + Anf) == 0:
        return 0
    if ((Akf + Anf) != 0) and ((Akp + Anp) == 0):
        return 1
    return 1 / ((Akf / (Anf + Akf)) + (Akp / (Anp + Akp)))


def turantula_sub_two(Akf, Anf, Akp, Anp):
    if (Akf + Akp) == 0 or (Akf + Anf) == 0:
        return 0
    if ((Akf + Anf) != 0) and ((Akp + Anp) == 0):
        return 1
    return Akf / (Anf + Akf)


def turantula_sub_three(Akf, Anf, Akp, Anp):
    if (Akf + Akp) == 0 or (Akf + Anf) == 0:
        return 0
    if ((Akf + Anf) != 0) and ((Akp + Anp) == 0):
        return 1
    return 1 / (Anf + Akf)


def naish1(Akf, Anf, Akp, Anp):

    return -1 if Anf > 0 else Anp


def binary(Akf, Anf, Akp, Anp):
    return 0 if Akf > 0 else 1


def dstar2(Akf, Anf, Akp, Anp):
    if (Akp + Anf) == 0:
        return sys.float_info.max
    return (math.pow(Akf, 2)) / (Akp + Anf)


def crosstab(Akf, Akp, Anf, Anp):
    tf = Akf + Anf
    tp = Akp + Anp
    N = tf + tp
    Sw = 0
    Ncw = Akf + Akp
    Nuw = Anf + Anp
    try:
        Ecfw = Ncw * (tf / N)
        Ecsw = Ncw * (tp / N)
        Eufw = Nuw * (tf / N)
        Eusw = Nuw * (tp / N)
        X2w = pow(Akf - Ecfw, 2) / Ecfw + pow(Akp - Ecsw, 2) / Ecsw + pow(Anf - Eufw, 2) / Eufw + pow(
            Anp - Eusw, 2) / Eusw
        yw = (Akf / tf) / (Akp / tp)
        if yw > 1:
            Sw = X2w
        elif yw < 1:
            Sw = -X2w
    except:
        Sw = 0
    return Sw
# endregion

# 自动创建不存在的目录
def checkAndCreateDir(Path):
    if not os.path.exists(Path):
        os.mkdir(Path)


# 生成hmer.json
def countMBFLFunctionSus(project, version):
    try:
        with open(faultlocalizationResultPath + "/" + project + "/" + version + "/susFunction/hmer.json", "r") as f:
            complete = json.load(f)
        return complete["type4"]
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")
# 计算TopN
def calTopN(project, version, susResult, FileName):
    try:
        with open(os.path.join(faultlocalizationResultPath, project, version, 'falutFunction.json'), 'r') as f:
            faultLocalization = json.load(f)
        # 将真实错误函数信息拼接起来
        faultFunc = list()
        tmp = '<419>'
        for item in faultLocalization:
            for index in faultLocalization[item]:
                faultFunc.append(item[1:] + tmp + str(index))

        topNResultBest = dict()
        topNResultAverage = dict()
        topNResultWorst = dict()

        for method in mbflMethods:
            method_name = method.__name__
            # 将同一公式下的所有函数进行统一排序生成newSusResult
            newSusResult = dict()
            for item in susResult:
                for suskey, value in susResult[item][method_name].items():
                    k = item + tmp + str(suskey)
                    newSusResult[k] = value
            newSusResult = dict(sorted(newSusResult.items(), key=lambda item: item[1], reverse=True))

            for faultKey in faultFunc:
                key = '/' + faultKey.split('<419>')[0]
                line = faultKey.split('<419>')[1]
                # region 创建字典
                if topNResultBest.get(key) is None:
                    topNResultBest[key] = dict()
                if topNResultBest[key].get(line) is None:
                    topNResultBest[key][line] = dict()
                if topNResultAverage.get(key) is None:
                    topNResultAverage[key] = dict()
                if topNResultAverage[key].get(line) is None:
                    topNResultAverage[key][line] = dict()
                if topNResultWorst.get(key) is None:
                    topNResultWorst[key] = dict()
                if topNResultWorst[key].get(line) is None:
                    topNResultWorst[key][line] = dict()
                # endregion

                if newSusResult.get(faultKey) is None:
                    topNResultBest[key][line][method_name] = -1
                    topNResultAverage[key][line][method_name] = -1
                    topNResultWorst[key][line][method_name] = -1
                    continue

                faultSus = newSusResult[faultKey]

                startFlagIndex = -1
                repeatFaultTime = 0
                endFlagIndex = -1
                ind = 0
                for item, value in newSusResult.items():
                    ind += 1
                    if math.isnan(value):
                        continue
                    if value > faultSus:
                        continue
                    if value == faultSus:
                        if startFlagIndex == -1:
                            startFlagIndex = ind
                        else:
                            if item in faultFunc:
                                repeatFaultTime += 1
                    else:
                        endFlagIndex = ind - 1 - repeatFaultTime
                        break
                # 最好排名
                topNResultBest[key][line][method_name] = startFlagIndex
                # 平均排名
                if endFlagIndex == -1:
                    endFlagIndex = ind
                if startFlagIndex == -1 or endFlagIndex == -1:
                    topNResultAverage[key][line][method_name] = -1
                else:
                    topNResultAverage[key][line][method_name] = (startFlagIndex + endFlagIndex) / 2
                # 最坏排名
                topNResultWorst[key][line][method_name] = endFlagIndex

        checkAndCreateDir(os.path.join(faultlocalizationResultPath, project, version, "topNFunctionBest"))
        checkAndCreateDir(os.path.join(faultlocalizationResultPath, project, version, "topNFunctionAverage"))
        checkAndCreateDir(os.path.join(faultlocalizationResultPath, project, version, "topNFunctionWorst"))
        with open(os.path.join(faultlocalizationResultPath, project, version, "topNFunctionBest", FileName), 'w') as f:
            f.write(json.dumps(topNResultBest, indent=2))
            f.close()
        with open(os.path.join(faultlocalizationResultPath, project, version, "topNFunctionAverage", FileName),
                  'w') as f:
            f.write(json.dumps(topNResultAverage, indent=2))
            f.close()
        with open(os.path.join(faultlocalizationResultPath, project, version, "topNFunctionWorst", FileName), 'w') as f:
            f.write(json.dumps(topNResultWorst, indent=2))
            f.close()

        return topNResultBest, topNResultAverage, topNResultWorst
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')

# 计算MFR MAR
def calMFRandMAR(project, version, topNResult, FileName):
    try:
        MFR = dict()
        MAR = dict()
        totalFunc = 0
        with open("../../d4j/outputClean/" + project + "/" + version + "/FunctionList.txt", 'r') as f:
            FunctionList = f.readlines()
        cnt = 0
        for key in topNResult.keys():
            totalFuncKey = key.split('/')
            totalFuncKey = '/'.join(totalFuncKey[-3:])
            for func in FunctionList:
                if totalFuncKey in func:
                    totalFunc = totalFunc + 1
                elif totalFunc!=0:
                    break
            for line in topNResult[key].keys():
                cnt = cnt + 1
                for method, value in topNResult[key][line].items():
                    if value == -1:
                        value = totalFunc
                    # 统计MFR
                    if MFR.get(method) is None:
                        MFR[method] = value
                    if MFR[method] > value:
                        MFR[method] = value
                    # 统计MAR
                    if MAR.get(method) is None:
                        MAR[method] = 0
                    MAR[method] += value
        for method in MAR.keys():
            MAR[method] = MAR[method] / cnt

        checkAndCreateDir(os.path.join(faultlocalizationResultPath, project, version, "MFR"))
        checkAndCreateDir(os.path.join(faultlocalizationResultPath, project, version, "MAR"))
        with open(os.path.join(faultlocalizationResultPath, project, version, "MFR", FileName), 'w') as f:
            f.write(json.dumps(MFR, indent=2))
            f.close()
        with open(os.path.join(faultlocalizationResultPath, project, version, "MAR", FileName), 'w') as f:
            f.write(json.dumps(MAR, indent=2))
            f.close()

        return MFR, MAR
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
# 读取sbfl.json
def countSBFLFunctionSus(project, version):
    try:
        with open(faultlocalizationResultPath + "/" + project + "/" + version + "/susFunction/sbfl.json", "r") as f:
            sbfl = json.load(f)
        return sbfl
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")


# 计数
def countTonN(root_path, project, mode):
    """
    遍历指定目录下所有子目录是否存在指定文件,如果存在则使用json.load读取文件内容
    """
    projectStatistics = dict()
    print(project,mode.split('Function')[1])
    for version in os.listdir(root_path + "/" + project):
        if not version.endswith('b'):
            continue
        if not failVersion.get(project) is None and version in failVersion[project]:
            continue
        print(version)
        for root, dirs, files in os.walk(root_path + "/" + project + "/" + version + "/" + mode):
            # 遍历当前目录下的所有文件
            for filename in files:
                if filename != 'hmer.json':
                    continue
                if filename not in projectStatistics:
                    projectStatistics[filename] = dict()
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                for key in data.keys():
                    for line in data[key].keys():
                        for item in data[key][line].keys():
                            if item not in projectStatistics[filename]:
                                projectStatistics[filename][item] = Counter({'top1': 0, 'top2': 0, 'top3': 0, 'top4': 0, 'top5': 0, 'top10': 0})
                            if data[key][line][item] <= 10 and data[key][line][item] >= 0:
                                projectStatistics[filename][item]['top10'] += 1
                            if data[key][line][item] <= 5 and data[key][line][item] >= 0:
                                projectStatistics[filename][item]['top5'] += 1
                            if data[key][line][item] <= 4 and data[key][line][item] >= 0:
                                projectStatistics[filename][item]['top4'] += 1
                            if data[key][line][item] <= 3 and data[key][line][item] >= 0:
                                projectStatistics[filename][item]['top3'] += 1
                            if data[key][line][item] <= 2 and data[key][line][item] >= 0:
                                projectStatistics[filename][item]['top2'] += 1
                            if data[key][line][item] <= 1 and data[key][line][item] >= 0:
                                projectStatistics[filename][item]['top1'] += 1
    checkAndCreateDir('./topN')
    checkAndCreateDir('./topN/' + mode)
    with open("./topN/" + mode + '/' + project, 'w') as f:
        f.write(json.dumps(projectStatistics, indent=2))

def countMAR(root_path, project):
    projectStatistics = dict()
    count = 0
    for version in os.listdir(root_path + "/" + project):
        if not failVersion.get(project) is None and version in failVersion[project]:
            continue
        if os.path.exists(root_path + "/" + project + "/" + version + "/MAR"):
            count += 1
        for root, dirs, files in os.walk(root_path + "/" + project + "/" + version + "/MAR"):
            # 遍历当前目录下的所有文件
            for filename in files:
                if filename not in projectStatistics:
                    projectStatistics[filename] = dict()
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                for key in data.keys():
                    if projectStatistics[filename].get(key) is None:
                        projectStatistics[filename][key] = 0
                    projectStatistics[filename][key] += data[key]
    
    for filename in projectStatistics.keys():
        for key in projectStatistics[filename].keys():
            projectStatistics[filename][key] /= count
    print('MAR Count',count)
    # print(projectStatistics)
    checkAndCreateDir('./MAR/')
    with open("./MAR/" + project, 'w') as f:
        f.write(json.dumps(projectStatistics, indent=2))
    
    return

def countMFR(root_path, project):
    projectStatistics = dict()
    count = 0
    for version in os.listdir(root_path + "/" + project):
        if not failVersion.get(project) is None and version in failVersion[project]:
            continue
        if os.path.exists(root_path + "/" + project + "/" + version + "/MFR"):
            count += 1
        for root, dirs, files in os.walk(root_path + "/" + project + "/" + version + "/MFR"):
            # 遍历当前目录下的所有文件
            for filename in files:
                if filename not in projectStatistics:
                    projectStatistics[filename] = dict()
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                for key in data.keys():
                    if projectStatistics[filename].get(key) is None:
                        projectStatistics[filename][key] = 0
                    projectStatistics[filename][key] += data[key]
    
    for filename in projectStatistics.keys():
        for key in projectStatistics[filename].keys():
            projectStatistics[filename][key] /= count
    print('MFR Count',count)
    # print(projectStatistics)
    checkAndCreateDir('./MFR/')
    with open("./MFR/" + project, 'w') as f:
        f.write(json.dumps(projectStatistics, indent=2))
    
    return

# 存储项目名称及其版本数
projectList = {
    "Chart": 26,
    # "Cli": 39,
    "Closure": 176,
    # "Codec": 18,
    # # "Collections": 4,
    # "Compress": 47,
    # "Csv": 16,
    # "Gson": 18,
    # "JacksonCore": 26,
    # "JacksonDatabind": 112,
    # "JacksonXml": 6,
    # "Jsoup": 93,
    # "JxPath": 22,
    "Lang": 65,
    "Math": 106,
    # "Mockito": 38,
    "Time": 27,
}
mbflMethods = [
    dstar
    # ,dstar_sub_one
    , ochiai
    # ,ochiai_sub_one
    # ,ochiai_sub_two
    , gp13
    # ,gp13_sub_one
    # ,gp13_sub_two
    , op2
    # ,op2_sub_one
    # ,op2_sub_two
    , jaccard
    # ,jaccard_sub_one
    , russell
    # ,russell_sub_one
    , turantula
    # ,turantula_sub_one
    , naish1
    , binary
    , crosstab
    , dstar2
]
if __name__ == "__main__":
    with open("./failVersion.json", "r") as f:
        failVersion = json.load(f)

    
    for project in projectList.keys():
        # 统计topN MFR MAR 
        for versionNum in range(1, projectList[project] + 1):
            version = str(versionNum) + 'b'
            if not failVersion.get(project) is None and version in failVersion[project]:
                continue
            print(f"{project} {version}")
            # 计算Hmer
            susResult = countMBFLFunctionSus(project,version)
            topNResultBest,topNResultAverage,topNResultWorst = calTopN(project, version, susResult, "hmer.json")
            MFR,MAR = calMFRandMAR(project, version, topNResultWorst, "hmer.json")
        
        # 计数
        countTonN(faultlocalizationResultPath, project, 'topNFunctionBest')
        countTonN(faultlocalizationResultPath, project, 'topNFunctionAverage')
        countTonN(faultlocalizationResultPath, project, 'topNFunctionWorst')
        print(f'{project} TOP N finish')
        countMAR(faultlocalizationResultPath, project)
        print(f'{project} MAR finish')
        countMFR(faultlocalizationResultPath, project)
        print(f'{project} MFR finish')