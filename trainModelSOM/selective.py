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


def logger_config(log_path):
    '''
    配置log
    :param log_path: 输出log路径
    :return:
    '''
    '''
    logger是日志对象,handler是流处理器,console是控制台输出(没有console也可以,将不会在控制台输出,会在日志文件中输出)
    '''
    # 获取logger对象
    logger = logging.getLogger()
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    # logger.addHandler(console)
    return logger


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

def countFunctionSus(project, version):
    try:
        with open("../../d4j/hugeToFunction/" + project + "/" + version + "/HugetoFunction.in", 'rb') as f:
            hugeToFunction = pickle.load(f)
        with open("../../d4j/outputClean/" + project + "/" + version + "/FunctionList.txt", 'r') as f:
            FunctionList = f.readlines()
        with open("../../d4j/outputClean/" + project + "/" + version + "/HugeToFile.txt", 'r') as f:
            hugeToFile = f.readlines()

        hugeToFiledict = dict()
        for i in range(0, len(hugeToFile)):
            if hugeToFiledict.get(hugeToFile[i].split("\t")[0]) == None:
                hugeToFiledict[hugeToFile[i].split("\t")[0]] = dict()
            functionLine = hugeToFunction[i] + 1
            count = sum(1 for element in FunctionList[0:functionLine] if
                        FunctionList[functionLine - 1].split(":")[0] in element)
            hugeToFiledict[hugeToFile[i].split("\t")[0]][hugeToFile[i].split("\t")[1].strip()] = count

        suspiciousPath = faultlocalizationResultPath + "/" + project + "/" + version + "/susStatement"
        for root, dirs, files in os.walk(suspiciousPath):
            for file in files:
                checkAndCreateDir(faultlocalizationResultPath + "/" + project + "/" + version + "/susFunction")
                functionSus = dict()
                file_path = os.path.join(root, file)
                # if os.path.exists(faultlocalizationResultPath + "/" + project + "/" + version + "/susFunction/" + file):
                #     continue
                if file != 'selective.json':
                    continue
                with open(file_path, 'r') as f:
                    sus = json.load(f)
                for j in range(1, 5):
                    functionSus[f'type{j}'] = dict()
                    for key in sus[f'type{j}'].keys():
                        functionSus[f'type{j}'][key] = dict()
                        for method in sus[f'type{j}'][key].keys():
                            functionSus[f'type{j}'][key][method] = dict()
                            for line in sus[f'type{j}'][key][method].keys():
                                for k in hugeToFiledict.keys():
                                    if k in key:
                                        break
                                if hugeToFiledict[k].get(str(int(line) - 1)) == None:
                                    continue
                                count = hugeToFiledict[k][str(int(line) - 1)]
                                if functionSus[f'type{j}'][key][method].get(count) == None:
                                    functionSus[f'type{j}'][key][method][count] = sus[f'type{j}'][key][method][line]
                with open(faultlocalizationResultPath + "/" + project + "/" + version + "/susFunction/" + file,
                          'w') as f:
                    f.write(json.dumps(functionSus, indent=2))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f'\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m')
        logging.error(f'Error in {file_name} at line {line_number}: {e}')


def calFomMbfl(project, version, muInfoList, resultList):
    """
    通过变异体的执行矩阵和杀死矩阵计算语句怀疑度
    :param Fom: 变异体的信息,主要用到行号
    :param FomResult: 变异体的执行结果和杀死信息,数组形式存储,第一个是执行结果,第二个是杀死信息.
                      执行结果: 1代表失败,0代表通过
                      杀死信息: 1代表杀死,0代表没杀死
    :return: 变异体信息列表
    """
    try:
        suspiciousFirstOrderPath = os.path.join(
            faultlocalizationResultPath,
            project,
            version,
            "susStatement",
            "selective.json",
        )
        susResult = {}
        susFOM = {}
        for j in range(1, 5):
            susResult[f"type{j}"] = dict()
            susFOM[f"type{j}"] = dict()
            for method in mbflMethods:
                susFOM[f"type{j}"][str(method).split(" ")[1]] = list()
            if resultList == None:
                resultList = []
            for i in range(0, len(resultList)):
                if resultList[i]["status"] == 0:
                    susFOM[f"type{j}"][str(method).split(" ")[1]].append(0)
                    continue
                Anp = 0
                Anf = 0
                Akp = 0
                Akf = 0
                if susResult[f"type{j}"].get(muInfoList[i]["relativePath"]) == None:
                    susResult[f"type{j}"][muInfoList[i]["relativePath"]] = dict()
                for index in range(0, len(resultList[i]["passList"][f"type{j}"])):
                    if resultList[i]["passList"][f"type{j}"][index] == 1:
                        if resultList[i]["killList"][f"type{j}"][index] == 1:
                            Akf += 1
                        else:
                            Anf += 1
                    else:
                        if resultList[i]["killList"][f"type{j}"][index] == 1:
                            Akp += 1
                        else:
                            Anp += 1
                for method in mbflMethods:
                    susFOM[f"type{j}"][str(method).split(" ")[1]].append(
                        method(Akf, Anf, Akp, Anp)
                    )
                    if (
                        susResult[f"type{j}"][muInfoList[i]["relativePath"]].get(
                            str(method).split(" ")[1]
                        )
                        == None
                    ):
                        susResult[f"type{j}"][muInfoList[i]["relativePath"]][
                            str(method).split(" ")[1]
                        ] = dict()
                    if (
                        susResult[f"type{j}"][muInfoList[i]["relativePath"]][
                            str(method).split(" ")[1]
                        ].get(resultList[i]["linenum"])
                        == None
                    ):
                        susResult[f"type{j}"][muInfoList[i]["relativePath"]][
                            str(method).split(" ")[1]
                        ][resultList[i]["linenum"]] = method(Akf, Anf, Akp, Anp)
                    else:
                        susResult[f"type{j}"][muInfoList[i]["relativePath"]][
                            str(method).split(" ")[1]
                        ][resultList[i]["linenum"]] = max(
                            susResult[f"type{j}"][muInfoList[i]["relativePath"]][
                                str(method).split(" ")[1]
                            ][resultList[i]["linenum"]],
                            method(Akf, Anf, Akp, Anp),
                        )
                for item in susResult[f"type{j}"].keys():
                    for method in mbflMethods:
                        susResult[f"type{j}"][item][str(method).split(" ")[1]] = dict(
                            sorted(
                                susResult[f"type{j}"][item][
                                    str(method).split(" ")[1]
                                ].items(),
                                key=operator.itemgetter(1),
                                reverse=True,
                            )
                        )

        checkAndCreateDir(os.path.join(faultlocalizationResultPath, project))
        checkAndCreateDir(os.path.join(faultlocalizationResultPath, project, version))
        checkAndCreateDir(
            os.path.join(faultlocalizationResultPath, project, version, "susStatement")
        )
        with open(suspiciousFirstOrderPath, "w") as f:
            f.write(json.dumps(susResult, indent=2))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")


def getInfo(project, version):
    faultLocalizationPath = os.path.join(faultlocalizationResultPath, project, version, "falutFunction.json")
    muInfo = os.path.join(faultlocalizationResultPath, project, version, "muInfo.json")
    muResult = os.path.join(faultlocalizationResultPath, project, version, "muResult.json")
    with open(faultLocalizationPath, "r") as f:
            faultLineDic = json.load(f)
    with open(muInfo, "r") as f:
            muInfoList = json.load(f)
    with open(muResult, "r") as f:
            resultList = json.load(f)
    return faultLineDic, muInfoList, resultList


def SELECTIVE(project, version):
    try:
        faultLineDic, muInfo, resultList = getInfo(project, version)
        
        mutationOperator = ['ORU', 'LOR', 'COR', 'LVR', 'STD', 'SOR', 'ROR', 'AOR']
        mutationOperator = random.sample(mutationOperator, 3)

        newInfoList = []
        for item in muInfo:
            if item['typeOp'] in mutationOperator:
                newInfoList.append(item)
        
        mutList = []
        for item in resultList:
            for info in newInfoList:
                if int(item['index']) == int(info['index']):
                    mutList.append(item)
                    break
        print(f'selective nums: {len(mutList)}')
        
        calFomMbfl(project, version, newInfoList, mutList)
        
        countFunctionSus(project, version)
        logging.info(f'success {project} {version}')

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")


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
    # 打印当前使用的python环境路径
    print(sys.executable)
    logger = logger_config(log_path='selective.log')
    pool = multiprocessing.Pool(processes=13)
    with open("./failVersion.json", "r") as f:
        failVersion = json.load(f)
    for project in projectList.keys():
        for versionNum in range(1, projectList[project] + 1):
            version = str(versionNum) + 'b'
            if not failVersion.get(project) is None and version in failVersion[project]:
                continue
            print(datetime.datetime.now())
            print(f"{project} {version} start")

            logging.info(f"{project} {version} start")
            # 最好不要超过18
            while pool._taskqueue.qsize() > 9:
                print(f"Waiting tasks in queue: {pool._taskqueue.qsize()}")
                time.sleep(15)
            pool.apply_async(SELECTIVE, (project,version))
            
    pool.close()
    pool.join()
    logging.info("all finish")
    exit(1)
