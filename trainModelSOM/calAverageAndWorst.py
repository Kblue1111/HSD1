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

# mbfl计算公式
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
# 自动创建不存在的目录
def checkAndCreateDir(Path):
    if not os.path.exists(Path):
        os.mkdir(Path)
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
def count_lines(filename):
    wc_output = subprocess.check_output(['wc', '-l', filename])
    line_count = int(wc_output.split()[0])
    return line_count

def calTopNMbflAverage(project, version, susResult, FileName, FaultFile, dir):
    try:
        topNResult = dict()
        with open(os.path.join(faultlocalizationResultPath, project, version, FaultFile), 'r')as f:
            faultLocalization = json.load(f)

        for key in faultLocalization.keys():
            topNResult[key] = dict()
            
            for line in faultLocalization[key]:
                topNResult[key][line] = dict()
                f = key[1:]
                if susResult.get(f) is None:
                    for method in mbflMethods:
                        topNResult[key][line][str(method).split(" ")[1]] = -1
                    continue
                for method in susResult[f].keys():
                    
                    susOfFaultStatement={}
                    for i in range(len(faultLocalization[key])):
                        if susResult[f][method].get(str(faultLocalization[key][i])) == None:
                            susOfFaultStatement[faultLocalization[key][i]] = -math.inf
                        else:
                            susOfFaultStatement[faultLocalization[key][i]] = susResult[f][method][str(faultLocalization[key][i])]
                    startFlagIndex=-1
                    repeatFaultTime=0
                    endFlagIndex=-1
                    ind = 0
                    # print(susOfFaultStatement)
                    for item, value in susResult[f][method].items():
                        # print(value)
                        ind += 1
                        # print(ind)
                        if math.isnan(value):
                            continue
                        if value > susOfFaultStatement[line]:
                            continue
                        if value == susOfFaultStatement[line]:
                            if startFlagIndex == -1:
                                startFlagIndex = ind
                            else:
                                if int(item) in faultLocalization[key]:
                                    repeatFaultTime += 1
                        else:
                            endFlagIndex = ind-1-repeatFaultTime
                            break
                    # if startFlagIndex == -1:
                    #     startFlagIndex = ind - 1 - repeatFaultTime
                    # if endFlagIndex == -1:
                    #     endFlagIndex = ind - 1 - repeatFaultTime
                    # print(startFlagIndex, endFlagIndex)
                    if startFlagIndex == -1 or endFlagIndex == -1:
                        topNResult[key][line][method] = -1
                    else:
                        topNResult[key][line][method] = (startFlagIndex+endFlagIndex)/2
                    # firstNum = -1
                    # lastNum = -1
                    # t = -1
                    # flag = True
                    # flagg = False
                    # for item, value in susResult[f][method].items():
                    #     if 
                    #     lastNum += 1
                    #     if flagg and value != t:
                    #         # topNResult[key][line][method] = lastNum
                    #         break
                    #     if value != t:
                    #         firstNum = lastNum
                    #         t = value
                    #     if int(item) == line:
                    #         # topNResult[key][line][method] = firstNum
                    #         flag = False
                    #         flagg = True

                    # # print(FileName, value, t)
                    # # print(firstNum, lastNum)
                    # if flag:
                    #     topNResult[key][line][method] = -1
                    # else:
                    #     # topNResult[key][line][method] = lastNum
                    #     topNResult[key][line][method] = (firstNum + lastNum) /2

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, dir))
        with open(os.path.join(faultlocalizationResultPath, project, version, dir, FileName), 'w') as f:
            f.write(json.dumps(topNResult, indent=2))
        return topNResult
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
def calMFRMbfl(project, version, topNResult, FileName, FaultFile, dir):
    try:
        faultFileLine = list()
        with open(os.path.join(faultlocalizationResultPath, project, version, FaultFile), 'r')as f:
            faultLocalization = json.load(f)
            for key in faultLocalization.keys():
                # print(project, version, os.path.join(djSrcPath, project, version, key[1:]))
                faultFileLine.append(count_lines(
                    os.path.join(djSrcPath, project, version, key[1:])))

        MFRResult = dict()
        for key in topNResult.keys():
            for line, methods in topNResult[key].items():
                for method, value in methods.items():
                    if MFRResult.get(method) == None:
                        MFRResult[method] = faultFileLine[0]
                    if value != -1 and MFRResult[method] > value:
                        MFRResult[method] = value

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, dir))
        with open(os.path.join(faultlocalizationResultPath, project, version, dir, FileName), 'w') as f:
            f.write(json.dumps(MFRResult, indent=2))

        f.close()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


def calMARMbfl(project, version, topNResult, FileName, FaultFile, dir):
    try:
        MARResult = dict()
        len = 0
        for key in topNResult.keys():
            for line, methods in topNResult[key].items():
                len += 1
                for method, value in methods.items():
                    if MARResult.get(method) == None:
                        MARResult[method] = 0
                    if value == -1:
                        MARResult[method] += count_lines(os.path.join(djSrcPath, project, version, key[1:]))
                    else:
                        MARResult[method] += value
        for key in MARResult.keys():
            MARResult[key] = MARResult[key]/len

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, dir))
        with open(os.path.join(faultlocalizationResultPath, project, version, dir, FileName), 'w') as f:
            f.write(json.dumps(MARResult, indent=2))

        f.close()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
projectList = {
    "Chart": 26,
    "Cli": 39,
    "Closure": 176,
    "Codec": 18,
    # "Collections": 4,
    "Compress": 47,
    "Csv": 16,
    "Gson": 18,
    "JacksonCore": 26,
    "JacksonDatabind": 112,
    "JacksonXml": 6,
    "Jsoup": 93,
    "JxPath": 22,
    "Lang": 65,
    "Math": 106,
    "Mockito": 38,
    "Time": 27,
}

if __name__ == "__main__":
    faultlocalizationResultPath = "/home/fanluxi/pmbfl/faultlocalizationResult"
    ablationPath = "/home/fanluxi/pmbfl/ablation"
    for project in projectList.keys():
        # project = "Chart"
        for versionNum in range(1, projectList[project] + 1):
            version = str(versionNum) + 'b'
            # version = "7b"
            susPath = os.path.join(
                faultlocalizationResultPath, project, version, "susFunction")
            topNPath = os.path.join(
                faultlocalizationResultPath, project, version, "topNFunction")
            for root, dirs, files in os.walk(susPath):
                for filename in files:
                    # if filename != "complete.json":
                    #     continue
                    print(project, version)
                    with open(susPath + "/" + filename, 'r') as f:
                        susResult = json.load(f)
                    topNResult = calTopNMbflAverage(
                        project, version, susResult, filename, "falutFunction.json", "topNFunction")
                    # print(topNResult)
                    calMFRMbfl(project, version, topNResult, filename, "falutFunction.json", "MFRFunction")
                    calMARMbfl(project, version, topNResult, filename, "falutFunction.json", "MARFunction")
                    # time.sleep(2)
                    # exit(1)