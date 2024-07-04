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

def countTests(project, version):
    nodeleteLine = list()
    with open(os.path.join(allTestsPath, project, version, 'all_tests'), 'r') as f:
        allTests = [line.strip() for line in f.readlines()]
    with open(os.path.join(outputCleanPath, project, version, 'all_tests.txt'), 'r') as f:
        allTestsClean = [line.strip() for line in f.readlines()]
    for i, item in enumerate(allTestsClean):
        item = item.split('#')[1] + "(" + item.split('#')[0] + ")"
        if item in allTests:
            nodeleteLine.append(i)
    nodeleteLine = list(range(len(allTestsClean)))
    return len(nodeleteLine)

def countLOC(project, version):
    filename = os.path.join(outputCleanPath, project, version, 'HugeToFile.txt')
    result = subprocess.run(['wc', '-l', filename], capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    line_count = int(output.split()[0])  
    return line_count

def countFaults(project, version):
    cnt = 0
    with open(os.path.join(faultlocalizationResultPath, project, version, 'falutFunction.json'), 'r') as f:
        falutFunction = json.load(f)
    for key in falutFunction:
        cnt = cnt + len(falutFunction[key])
    return cnt

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
if __name__ == "__main__":
    # 打印当前使用的python环境路径
    print(sys.executable)
    logger = logger_config(log_path='countPrograms.log')
    with open("./failVersion.json", "r") as f:
        failVersion = json.load(f)

    sumLOC = 0
    sumTests = 0
    sumFaults = 0
    sumVer = 0
    for project in projectList.keys():
        numLOC = 0
        numTests = 0
        numFaults = 0
        numVer = 0
        for versionNum in range(1, projectList[project] + 1):
            version = str(versionNum) + 'b'
            if not failVersion.get(project) is None and version in failVersion[project]:
                continue
            numVer = numVer + 1
            numLOC = numLOC + countLOC(project, version)
            numTests = numTests + countTests(project, version)
            numFaults = numFaults + countFaults(project, version)
        logging.info(f"{project} {version} Versions: {numVer}")
        logging.info(f"{project} {version} LOCs: {numLOC}")
        logging.info(f"{project} {version} Tests: {numTests}")
        logging.info(f"{project} {version} Faults: {numFaults}")
        sumVer = numVer + sumVer
        sumLOC = numLOC + sumLOC
        sumTests = numTests + sumTests
        sumFaults = numFaults + sumFaults
    logging.info(f"Versions: {sumVer}")
    logging.info(f"LOCs: {sumLOC}")
    logging.info(f"Tests: {sumTests}")
    logging.info(f"Faults: {sumFaults}")
