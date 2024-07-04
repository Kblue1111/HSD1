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
SOMfaultlocalizationResultPath = configData['SOMfaultlocalizationResultPath']
djfeaturePath = configData['djfeaturePath']
djSrcPath = configData['djSrcPath']
outputCleanPath = configData['outputCleanPath']
dynamicFeaturePath = configData['dynamicFeaturePath']
tpydataPath = configData['tpydataPath']
allTestsPath = configData['allTestsPath']


def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    return s.getsockname()[0]


ip = get_host_ip()  # 获取当前ip


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
import re

# 根据单个变异体的行号获取在hugeCode的行号，将行号转化为标准形式
def getHugeLine(project, version, info):
    try:
        # 使用正则表达式进行分割
        split_result = re.split(f"{project}/{version}/\d+/", info['mutFilePath1'])
        # 获取分割后的第二部分
        if len(split_result) > 1:
            hugeCodeLineInfo1 = split_result[1]
            
        # 使用正则表达式进行分割
        split_result = re.split(f"{project}/{version}/\d+/", info['mutFilePath2'])
        # 获取分割后的第二部分
        if len(split_result) > 1:
            hugeCodeLineInfo2 = split_result[1]
        
        file_path = os.path.join(
            outputCleanPath, project, version, 'HugeToFile.txt')
        # Time 1b特殊处理，他的HugeToFile里面是  /time/Chronology.java	0 这种形式
        if project == 'Time' and version == '1b':
            hugeCodeLineInfo1 = '/time' + hugeCodeLineInfo1.split('time')[1]
            hugeCodeLineInfo2 = '/time' + hugeCodeLineInfo2.split('time')[1]
        line_num1 = -1
        line_num2 = -1
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
            for line_num, line in enumerate(lines, start=1):
                if line.strip() == f"{hugeCodeLineInfo1}\t{int(info['linenum1']) - 1}":
                    # return line_num
                    line_num1 = line_num
                if line.strip() == f"{hugeCodeLineInfo2}\t{int(info['linenum2']) - 1}":
                    # return line_num
                    line_num2 = line_num
        if line_num1 != -1 and line_num2 != -1:
            return line_num1, line_num2
        return False
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return False


# 根据变异体的信息和标准行号获取静态特征
def getStaticFeature(info, line_num, staticFea):
    try:
        staticFeature = list()
        # fcci + PMT 共 14 + 13个
        for i in range(0, len(staticFea['fcci'])):
            staticFeature.append(staticFea['fcci'][i][line_num - 1])
        for item in staticFea.keys():
            if item == 'fcci':
                continue
            staticFeature.append(staticFea[item][line_num - 1])
        return staticFeature
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None


# 获取某个版本的某个变异体的特征向量
def getFeatureVect(project, version, info, result, staticFea, dynamicFea, crFea, sfFea, ssFea):
    try:
        if result['status'] == 0:
            return False
        featureVect = list()
        line_num1, line_num2 = getHugeLine(project, version, info)
        if not line_num1 or not line_num2:
            print(f"\033[1;31mMutant {info['index']} HugeToFile not fund\033[0m")
            logging.error(f"mMutant {info['index']} HugeToFile not fund")
            return False
        staticFeature = getStaticFeature(info, line_num1, staticFea)
        operatorType = {'AOR': 0, 'LOR': 1, 'SOR': 2,
                        'COR': 3, 'ROR': 4, 'ORU': 5, 'LVR': 6, 'STD': 7}
        staticFeature.append(operatorType[info['typeOp1']])
        staticFeature.extend(getStaticFeature(info, line_num2, staticFea))
        staticFeature.append(operatorType[info['typeOp2']])
        # print(len(staticFeature))

        # 将list转化为numpy中的array，并转化成指定的形状
        staticFeature = np.array(staticFeature).reshape(-1, 28 * 2)
        dynamicFeature = np.array(dynamicFea).reshape(-1, 28)

        # 将staticFeature复制为(2199, 28)的二维数组
        staticFeature = np.tile(staticFeature, (dynamicFeature.shape[0], 1))
        # 将两个二维数组合并为(2199, 56)的二维数组
        featureVect = np.concatenate((staticFeature, dynamicFeature), axis=1).tolist()
        # print(len(featureVect[0]))
        # 加入60个测试用例特征   24 * 3 = 72
        for i, item in enumerate(featureVect):
            cr = []
            sf = []
            ss = []
            for key in crFea.keys():
                if key == 'dstar2':
                    continue
                cr.append(crFea[key][i])
                sf.append(sfFea[key][i])
                ss.append(ssFea[key][i])
            featureVect[i].extend(cr)
            featureVect[i].extend(sf)
            featureVect[i].extend(ss)
        # print(f'true or false: {myfeatureVect == featureVect}')

        # fcci:14 , pmt:14, 动态:28, 测试用例: 24 * 3=72 总共：14+14+28+72=128  fcci和pmt有4个重合的,14+14-4=24。 24+28+72=120
        # print(len(featureVect[0]))
        return featureVect  # [2199,128] 128= 56 + 24*3
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None


# 加载持久化文件
def loadFile(project, version):
    try:
        nodeleteLine = list()
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "subSOMInfo.json"), 'r') as f:
            muInfo = json.load(f)
        # muResult 的结果不正确，需要改正
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "somResult.json"), 'r') as f:
            muResults = json.load(f)
        with open(os.path.join(allTestsPath, project, version, 'all_tests'), 'r') as f:
            allTests = [line.strip() for line in f.readlines()]
        with open(os.path.join(outputCleanPath, project, version, 'all_tests.txt'), 'r') as f:
            allTestsClean = [line.strip() for line in f.readlines()]
        for i, item in enumerate(allTestsClean):
            item = item.split('#')[1] + "(" + item.split('#')[0] + ")"
            if item in allTests:
                nodeleteLine.append(i)
        # 有些版本allTests 和 allTestsClean 不一致，导致nodeleteLine和allTestsClean数量不一致，引发后面变异体执行结果有误报错，先临时处理
        nodeleteLine = list(range(len(allTestsClean)))
        with open(os.path.join(djfeaturePath, project, version, "static_fea"), 'rb') as f:
            staticFea = pickle.load(f)
        with open(os.path.join(dynamicFeaturePath, project, version, "static_all"), 'rb') as f:
            dynamicFea = pickle.load(f)
            dynamicFea = [dynamicFea[i] for i in range(len(dynamicFea)) if i in nodeleteLine]
        with open(os.path.join(tpydataPath, project, version, 'CR'), 'rb') as f:
            crFea = pickle.load(f)
            for key in crFea.keys():
                crFea[key] = [crFea[key][i] for i in range(len(crFea[key])) if i in nodeleteLine]
        with open(os.path.join(tpydataPath, project, version, 'SF'), 'rb') as f:
            sfFea = pickle.load(f)
            for key in sfFea.keys():
                sfFea[key] = [sfFea[key][i] for i in range(len(sfFea[key])) if i in nodeleteLine]
        with open(os.path.join(tpydataPath, project, version, 'SS'), 'rb') as f:
            ssFea = pickle.load(f)
            for key in ssFea.keys():
                ssFea[key] = [ssFea[key][i] for i in range(len(ssFea[key])) if i in nodeleteLine]

        # 这块的序号没问题吗，可能和上面的nodeleteLine不一致
        # nodeleteLine = list()
        # for i, item in enumerate(allTests):
        #     item = item.split('(')[1].split(')')[0] + "#" + item.split('(')[0]
        #     if item in allTestsClean:
        #         nodeleteLine.append(i)
        return muInfo, muResults, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None, None, None, None, None, None, None, None


def remoteFileExists(fileName):
    remote_host = '202.4.157.19'  # 远程Linux系统的IP地址
    remote_path = '/home/fanluxi/data/' + fileName  # 远程二进制文件的路径
    # print(remote_path)
    sshpass_cmd = f'sshpass -p "123456" ssh -o StrictHostKeyChecking=no fanluxi@{remote_host} "test -e {remote_path} && echo exists"'
    # print(sshpass_cmd)
    ssh = subprocess.Popen(sshpass_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = ssh.communicate()
    # print(stdout)
    if 'exists' in str(stdout):
        return True
    else:
        return False


def loadRemoteFile(fileName):
    try:
        remote_host = '202.4.157.19'  # 远程Linux系统的IP地址
        remote_path = '/home/fanluxi/data/' + fileName  # 远程二进制文件的路径
        local_path = fileName.replace('/', '_')  # 本地保存文件的路径

        # 执行SCP命令下载远程文件到本地
        sshpass_cmd = f'sshpass -p "123456" rsync -avz fanluxi@{remote_host}:{remote_path} {local_path}'
        # print(sshpass_cmd)
        subprocess.run(sshpass_cmd, shell=True, stdout=subprocess.DEVNULL)
        subprocess.run(['tar', '-xzvf', local_path], stdout=subprocess.DEVNULL)
        # 删除本地文件
        os.remove(local_path)

        # 加载本地文件并反序列化为Python对象
        with open("./featureVectList", 'rb') as f:
            featureVectList = pickle.load(f)
        # 删除本地文件
        os.remove("./featureVectList")

        # 加载本地文件并反序列化为Python对象
        with open("./passLabelList", 'rb') as f:
            passLabelList = pickle.load(f)
        # 删除本地文件
        os.remove("./passLabelList")

        # 加载本地文件并反序列化为Python对象
        with open("./killLabelList", 'rb') as f:
            killLabelList = pickle.load(f)
        # 删除本地文件
        os.remove("./killLabelList")

        # 加载本地文件并反序列化为Python对象
        with open("./museKillLabelList", 'rb') as f:
            museKillLabelList = pickle.load(f)
        # 删除本地文件
        os.remove("./museKillLabelList")

        # 加载本地文件并反序列化为Python对象
        with open("./pkMap", 'rb') as f:
            pkMap = pickle.load(f)
        # 删除本地文件
        os.remove("./pkMap")

        # 加载本地文件并反序列化为Python对象
        with open("./testNum", 'rb') as f:
            testNum = pickle.load(f)
        # 删除本地文件
        os.remove("./testNum")

        # 加载本地文件并反序列化为Python对象
        with open("./faultFileName", 'rb') as f:
            faultFileName = pickle.load(f)
        # 删除本地文件
        os.remove("./faultFileName")

        return np.array(featureVectList), np.array(passLabelList), np.array(killLabelList), np.array(
            museKillLabelList), pkMap, testNum, faultFileName
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


def saveRemoteFile(data, fileName):
    try:
        remote_host = '202.4.157.19'  # 远程Linux系统的IP地址
        remote_path = '/home/fanluxi/data/' + fileName  # 远程二进制文件的路径
        local_path = fileName.replace('/', '_')  # 本地保存文件的路径

        # 将Python对象序列化为二进制数据并保存到本地文件
        with open(local_path, 'wb') as f:
            pickle.dump(data, f)

        # 检查远程目录是否存在，如果不存在则创建
        dir_path, _ = os.path.split(remote_path)
        subprocess.run(['ssh', remote_host, f'mkdir -p {dir_path}'])

        # 执行rsync命令将本地文件上传到远程Linux系统
        subprocess.run(['rsync', '-avz', local_path, f'{remote_host}:{remote_path}'], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)

        # 删除本地文件
        os.remove(local_path)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


threadList = []


def process_muResults(muResults, project, version, muInfo, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine,error_flag):
    try:
        featureVectList = []
        passLabelList = []
        killLabelList = []
        museKillLabelList = []
        pkMap1 = []
        pkMap2 = []
        faultFileName1 = []
        faultFileName2 = []


        for i, muResult in enumerate(muResults):
            if error_flag.value:
                break
            # i = int(muResult['index']) - 1
            info = muResult
            # 如果passList和killList为空那么就是执行失败了  应该是在生成的时候解决一下
            if all(not info['passList']['type{}'.format(i + 1)] for i in range(4)) and all(
                    not info['killList']['type{}'.format(i + 1)] for i in range(4)):
                info['status'] = 0
            if info['status'] != 0:
                
                featureVect = getFeatureVect(
                    project, version, muInfo[i], info, staticFea, dynamicFea, crFea, sfFea, ssFea)
                result = []
                for j in range(len(info['passList'])):
                    if j in nodeleteLine:
                        result.append(info['passList']['type{}'.format(j + 1)])
                # 数量对不上
                if type(featureVect) is list:
                    # fanluxi
                    # if len([info['passList']['type{}'.format(j + 1)] for j in range(len(info['passList'])) if j in nodeleteLine][3]) != len(featureVect):
                    # XSQ  这块是这样的吗，4种类型要全加入进去吗，还是只加入type4
                    if len(result[3]) != len(featureVect):
                        print(f"\033[1;31m {project} {version} 变异体 {muResult['index']}执行结果有误！\033[0m")
                        logging.error(f"变异体{muResult['index']}执行结果有误！")
                        error_flag.value = True
                        break
                    else:
                        featureVectList.extend(featureVect)  # [1805,128] -> [3610,128] 改为 [[1805,128],[1805,128]]
                        pkMap1.append(info['linenum1'])
                        pkMap2.append(info['linenum2'])
                        faultFileName1.append(muInfo[i]['relativePath1'])
                        faultFileName2.append(muInfo[i]['relativePath2'])
                        # 加4种type
                        # passLabelList.extend([info['passList']['type{}'.format(i + 1)] for i in range(
                        #     len(info['passList'])) if i in nodeleteLine])
                        # killLabelList.extend([info['killList']['type{}'.format(i + 1)] for i in range(
                        #     len(info['killList'])) if i in nodeleteLine])
                        # museKillLabelList.extend([info['mKillList'][i] for i in range(
                        #     len(info['mKillList'])) if i in nodeleteLine])

                        # 只加type4
                        passLabelList.extend(info['passList']['type3'])
                        killLabelList.extend(info['killList']['type3'])
                        museKillLabelList.extend([])
        return featureVectList, passLabelList, killLabelList, museKillLabelList, pkMap1, pkMap2, faultFileName1, faultFileName2
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')

import gzip


def load_compressed_data(project, version):
    if os.path.exists(f"../tmp/{project}_{version}_data.pkl.gz"):
        with gzip.open(f"../tmp/{project}_{version}_data.pkl.gz", "rb") as f:
            return pickle.load(f)
    else:
        return None

# 获取训练集和测试集数据  把128个特征拼成128维度的向量
def getTrainData(project, version):
    # 检查本地是否存在压缩数据文件，如果存在，则直接读取数据并返回
    compressed_data = load_compressed_data(project, version)
    if compressed_data:
        return compressed_data
    try:
        featureVectList = []
        passLabelList = []
        killLabelList = []
        museKillLabelList = []
        pkMap1 = []
        pkMap2 = []
        faultFileName1 = []
        faultFileName2 = []
        muInfo, muResults, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine = loadFile(project, version)
        
        if len(muInfo) != len(muResults):
            print(f"{project} {version} 变异体信息数量{len(muInfo)}不等于变异体执行结果数量{len(muResults)}")
            logging.error(f"{project} {version} 变异体信息数量{len(muInfo)}不等于变异体执行结果数量{len(muResults)}")
            return None, None, None, None, None, None, None, None, None
        
        for i in range(1,len(muResults)):
            testNum = len(muResults[i]['passList']['type3'])
            if testNum != 0:
                break

        num_workers = 18
        chunk_size = len(muResults) // (2 * num_workers)
        # 步长为0没有意义
        if chunk_size == 0:
            chunk_size = 1
        with multiprocessing.Manager() as manager:
            error_flag = manager.Value('i', False)
            with multiprocessing.Pool(num_workers) as pools:
                results = []
                for i in range(0, len(muResults), chunk_size):
                    chunk = muResults[i:i + chunk_size]
                    chunk_muInfo = muInfo[i: i+chunk_size]
                    results.append(pools.apply_async(process_muResults, args=(
                        chunk, project, version, chunk_muInfo, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine,
                        error_flag)))
                for result in results:
                    featureVect, passLabel, killLabel, museKillLabel, pk1, pk2, faultFile1, faultFile2 = result.get()
                    featureVectList.extend(featureVect)
                    passLabelList.extend(passLabel)
                    killLabelList.extend(killLabel)
                    museKillLabelList.extend(museKillLabel)
                    pkMap1.extend(pk1)
                    pkMap2.extend(pk2)
                    faultFileName1.extend(faultFile1)
                    faultFileName2.extend(faultFile2)
                if error_flag.value:
                    return None, None, None, None, None, None, None, None, None
                
                print("start")

                # 在函数执行完毕后，将数据持久化保存到本地文件中（压缩形式）
                data = (np.array(featureVectList), np.array(passLabelList), np.array(killLabelList), np.array(museKillLabelList), pkMap1, pkMap2, testNum, faultFileName1, faultFileName2)
                
                # 将数据存储在内存中
                compressed_data = pickle.dumps(data)
                
                # 将所有数据一次性写入到文件中
                with gzip.open(f"../tmp/{project}_{version}_data.pkl.gz", "wb") as f:
                    f.write(compressed_data)

                return data


                return np.array(featureVectList), np.array(passLabelList), np.array(killLabelList), np.array(museKillLabelList), pkMap1, pkMap2, testNum, faultFileName1, faultFileName2
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None, None, None, None, None, None, None, None, None


def data_split(muInde, testNum, featureVectList, labelList, train_ratio=0.1):
    labelInde = list()
    for item in muInde:
        for i in range(testNum):
            labelInde.append(item * testNum + i)
    indx = np.arange(len(featureVectList))
    # 将labelInde作为训练集，其余作为测试集
    train_indx = np.isin(indx, labelInde)
    X_train, y_train = np.array(featureVectList)[train_indx], np.array(labelList)[train_indx]
    X_test, y_test, test_indx = [], [], []
    for i in range(len(indx)):
        if not train_indx[i]:
            X_test.append(featureVectList[i])
            y_test.append(labelList[i])
            test_indx.append(indx[i])
    X_test, y_test, test_indx = np.array(X_test), np.array(y_test), np.array(test_indx)
    # 如果训练集数量不足10%，随机将测试集中的样本加入到训练集中
    if len(X_train) < train_ratio * len(featureVectList):
        # replace=False 的意思是不放回的随机抽样
        to_add = np.random.choice(test_indx, size=int(train_ratio * len(featureVectList)) - len(X_train), replace=False)
        # 拼接 200*128和200*128拼接后会变成400*128
        X_train = np.concatenate([X_train, X_test[np.isin(test_indx, to_add)]])
        y_train = np.concatenate([y_train, y_test[np.isin(test_indx, to_add)]])
        # 逻辑取反 np.logical_not 去掉被当成训练集的那些数据
        X_test = X_test[np.logical_not(np.isin(test_indx, to_add))]
        y_test = y_test[np.logical_not(np.isin(test_indx, to_add))]
        test_indx = test_indx[np.logical_not(np.isin(test_indx, to_add))]
        # 添加新增的训练集序号
        train_indx[to_add] = True

    # 分离训练和测试集
    # indx = np.arange(len(featureVectList))
    # X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(featureVectList, labelList, indx, train_size=train_ratio,random_state=419, shuffle=True)
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'indx_train': train_indx,
        'indx_test': test_indx
    }
    return data

# region 模型
def randomForestModel(featureVectList, labelList, saveFile, project, version, mode, kill_data, i):
    try:
        X_train = kill_data['X_train']
        X_test = kill_data['X_test']
        y_train = kill_data['y_train']
        y_test = kill_data['y_test']
        indx_train = kill_data['indx_train']
        indx_test = kill_data['indx_test']

        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project, version, "Model"))

        if os.path.isfile(os.path.join(SOMfaultlocalizationResultPath, project, version, "Model", mode + f"_pmbfl_{i}_2Model")):
            rf_clf = loadModel(os.path.join(SOMfaultlocalizationResultPath, project, version, "Model", mode + f"_pmbfl_{i}_2Model"))
        else:
            # 定义随机森林分类器
            # rf_clf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1)
            rf_clf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, n_jobs=-1)
            rf_clf.fit(X_train, y_train)

            if saveFile:
                saveModel(rf_clf, os.path.join(SOMfaultlocalizationResultPath, project, version, "Model", mode + f"_pmbfl_{i}_2Model"))

        # 用分类器预测测试集的结果
        y_pred = rf_clf.predict(X_test)
        # 将指标存储为字典  评价的是模型的指标
        results = {
            # 计算准确率
            "accuracy": accuracy_score(y_test, y_pred),
            # 计算精确度
            "precision": precision_score(y_test, y_pred),
            # 计算召回率
            "recall": recall_score(y_test, y_pred),
            # 计算F1得分
            "f1": f1_score(y_test, y_pred),
            # 计算ROC曲线下面积
            "auc": roc_auc_score(y_test, y_pred),
            # 计算预测误差
            "prediction_error": mean_absolute_error(y_test, y_pred)
        }

        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "ModelMetrics", mode + f"_pmbfl_{i}_2Model.json"),"w") as f:
            f.write(json.dumps(results, indent=2))

        test_results = {indx_test[i]: y_pred[i] for i in range(len(indx_test))}
        result = np.empty(len(featureVectList), dtype=y_train.dtype)
        result[indx_train] = y_train
        result[indx_test] = [test_results.get(i, 0) for i in indx_test]

        return result
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None


def svmModel(featureVectList, labelList, saveFile, project, version, mode, kill_data):
    try:
        X_train = kill_data['X_train']
        X_test = kill_data['X_test']
        y_train = kill_data['y_train']
        y_test = kill_data['y_test']
        indx_train = kill_data['indx_train']
        indx_test = kill_data['indx_test']

        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_svmModel")):
            svm_clf = loadModel(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_svmModel"))
        else:
            n_estimators = 10
            svm_clf = SVC(kernel='linear', C=0.1, cache_size=200, shrinking=True)
            clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'),
                                                        max_samples=1.0 / n_estimators, n_estimators=n_estimators))
            svm_clf.fit(X_train, y_train)

            if saveFile:
                saveModel(svm_clf, os.path.join(
                    SOMfaultlocalizationResultPath, project, version, "Model", mode + "_svmModel"))

        # 用预测测试集的结果
        y_pred = svm_clf.predict(X_test)

        # 将指标存储为字典
        results = {
            # 计算准确率
            "accuracy": accuracy_score(y_test, y_pred),
            # 计算精确度
            "precision": precision_score(y_test, y_pred),
            # 计算召回率
            "recall": recall_score(y_test, y_pred),
            # 计算F1得分
            "f1": f1_score(y_test, y_pred),
            # 计算ROC曲线下面积
            "auc": roc_auc_score(y_test, y_pred),
            # 计算预测误差
            "prediction_error": mean_absolute_error(y_test, y_pred)
        }

        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "ModelMetrics", mode + "_svmModel.json"),"w") as f:
            f.write(json.dumps(results, indent=2))

        test_results = {indx_test[i]: y_pred[i] for i in range(len(indx_test))}
        result = np.empty(len(featureVectList), dtype=y_train.dtype)
        result[indx_train] = y_train
        result[indx_test] = [test_results.get(i, 0) for i in indx_test]

        return result
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None


def logisticRegressionModel(featureVectList, labelList, saveFile, project, version, mode, kill_data):
    try:
        X_train = kill_data['X_train']
        X_test = kill_data['X_test']
        y_train = kill_data['y_train']
        y_test = kill_data['y_test']
        indx_train = kill_data['indx_train']
        indx_test = kill_data['indx_test']

        # 标准化特征向量
        scaler = Normalizer().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_lrModel")):
            lr_clf = loadModel(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_lrModel"))
        else:
            # 定义逻辑回归分类器
            lr_clf = LogisticRegression(solver='liblinear')
            lr_clf.fit(X_train_std, y_train)

            if saveFile:
                saveModel(lr_clf, os.path.join(
                    SOMfaultlocalizationResultPath, project, version, "Model", mode + "_lrModel"))

        # 用分类器预测测试集的结果
        y_pred = lr_clf.predict(X_test_std)

        # 将指标存储为字典
        results = {
            # 计算准确率
            "accuracy": accuracy_score(y_test, y_pred),
            # 计算精确度
            "precision": precision_score(y_test, y_pred),
            # 计算召回率
            "recall": recall_score(y_test, y_pred),
            # 计算F1得分
            "f1": f1_score(y_test, y_pred),
            # 计算ROC曲线下面积
            "auc": roc_auc_score(y_test, y_pred),
            # 计算预测误差
            "prediction_error": mean_absolute_error(y_test, y_pred)
        }

        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "ModelMetrics", mode + "_lrModel.json"),"w") as f:
            f.write(json.dumps(results, indent=2))

        test_results = {indx_test[i]: y_pred[i] for i in range(len(indx_test))}
        result = np.empty(len(featureVectList), dtype=y_train.dtype)
        result[indx_train] = y_train
        result[indx_test] = [test_results.get(i, 0) for i in indx_test]

        return result
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e} version {version} \033[0m')
        logging.error(f'Error at line {line_number}: {e} version {version} ')
        return None


def knnModel(featureVectList, labelList, saveFile, project, version, mode, kill_data):
    try:
        X_train = kill_data['X_train']
        X_test = kill_data['X_test']
        y_train = kill_data['y_train']
        y_test = kill_data['y_test']
        indx_train = kill_data['indx_train']
        indx_test = kill_data['indx_test']

        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_knnModel")):
            knn_clf = loadModel(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_knnModel"))
        else:
            knn_clf = KNeighborsClassifier(n_neighbors=3, leaf_size=10, weights='distance', algorithm='kd_tree',
                                           n_jobs=-1)
            knn_clf.fit(X_train_std, y_train)

            if saveFile:
                saveModel(knn_clf, os.path.join(
                    SOMfaultlocalizationResultPath, project, version, "Model", mode + "_knnModel"))

        # 用分类器预测测试集的结果
        y_pred = knn_clf.predict(X_test_std)

        # 将指标存储为字典
        results = {
            # 计算准确率
            "accuracy": accuracy_score(y_test, y_pred),
            # 计算精确度
            "precision": precision_score(y_test, y_pred),
            # 计算召回率
            "recall": recall_score(y_test, y_pred),
            # 计算F1得分
            "f1": f1_score(y_test, y_pred),
            # 计算ROC曲线下面积
            "auc": roc_auc_score(y_test, y_pred),
            # 计算预测误差
            "prediction_error": mean_absolute_error(y_test, y_pred)
        }

        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "ModelMetrics", mode + "_knnModel.json"),"w") as f:
            f.write(json.dumps(results, indent=2))

        test_results = {indx_test[i]: y_pred[i] for i in range(len(indx_test))}
        result = np.empty(len(featureVectList), dtype=y_train.dtype)
        result[indx_train] = y_train
        result[indx_test] = [test_results.get(i, 0) for i in indx_test]

        return result
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None


def naiveBayesModel(featureVectList, labelList, saveFile, project, version, mode, kill_data):
    try:
        X_train = kill_data['X_train']
        X_test = kill_data['X_test']
        y_train = kill_data['y_train']
        y_test = kill_data['y_test']
        indx_train = kill_data['indx_train']
        indx_test = kill_data['indx_test']

        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_nbModel")):
            nb_clf = loadModel(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_nbModel"))
        else:
            nb_clf = GaussianNB()
            nb_clf.fit(X_train_std, y_train)

            if saveFile:
                saveModel(nb_clf, os.path.join(
                    SOMfaultlocalizationResultPath, project, version, "Model", mode + "_nbModel"))

        # 用分类器预测测试集的结果
        y_pred = nb_clf.predict(X_test_std)

        # 将指标存储为字典
        results = {
            # 计算准确率
            "accuracy": accuracy_score(y_test, y_pred),
            # 计算精确度
            "precision": precision_score(y_test, y_pred),
            # 计算召回率
            "recall": recall_score(y_test, y_pred),
            # 计算F1得分
            "f1": f1_score(y_test, y_pred),
            # 计算ROC曲线下面积
            "auc": roc_auc_score(y_test, y_pred),
            # 计算预测误差
            "prediction_error": mean_absolute_error(y_test, y_pred)
        }

        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "ModelMetrics", mode + "_nbModel.json"),"w") as f:
            f.write(json.dumps(results, indent=2))

        test_results = {indx_test[i]: y_pred[i] for i in range(len(indx_test))}
        result = np.empty(len(featureVectList), dtype=y_train.dtype)
        result[indx_train] = y_train
        result[indx_test] = [test_results.get(i, 0) for i in indx_test]

        return result
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None


def mlpModel(featureVectList, labelList, saveFile, project, version, mode, kill_data):
    try:
        X_train = kill_data['X_train']
        X_test = kill_data['X_test']
        y_train = kill_data['y_train']
        y_test = kill_data['y_test']
        indx_train = kill_data['indx_train']
        indx_test = kill_data['indx_test']

        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_mlpModel")):
            mlp_clf = loadModel(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_mlpModel"))
        else:
            mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10000)
            mlp_clf.fit(X_train_std, y_train)

            if saveFile:
                saveModel(mlp_clf, os.path.join(
                    SOMfaultlocalizationResultPath, project, version, "Model", mode + "_mlpModel"))

        # 用分类器预测测试集的结果
        y_pred = mlp_clf.predict(X_test_std)

        # 将指标存储为字典
        results = {
            # 计算准确率
            "accuracy": accuracy_score(y_test, y_pred),
            # 计算精确度
            "precision": precision_score(y_test, y_pred),
            # 计算召回率
            "recall": recall_score(y_test, y_pred),
            # 计算F1得分
            "f1": f1_score(y_test, y_pred),
            # 计算ROC曲线下面积
            "auc": roc_auc_score(y_test, y_pred),
            # 计算预测误差
            "prediction_error": mean_absolute_error(y_test, y_pred)
        }

        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "ModelMetrics", mode + "_mlpModel.json"),"w") as f:
            f.write(json.dumps(results, indent=2))

        test_results = {indx_test[i]: y_pred[i] for i in range(len(indx_test))}
        result = np.empty(len(featureVectList), dtype=y_train.dtype)
        result[indx_train] = y_train
        result[indx_test] = [test_results.get(i, 0) for i in indx_test]

        return result
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None


def cnnModel(featureVectList, labelList, saveFile, project, version, mode, kill_data):
    try:
        X_train = kill_data['X_train']
        X_test = kill_data['X_test']
        y_train = kill_data['y_train']
        y_test = kill_data['y_test']
        indx_train = kill_data['indx_train']
        indx_test = kill_data['indx_test']

        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_cnnModel")):
            mlp_clf = loadModel(os.path.join(
                SOMfaultlocalizationResultPath, project, version, "Model", mode + "_cnnModel"))
        else:
            # # cnn
            # conv1_val = int(((119 - 3 + 1) - (3 - 1) - 1) / 3 + 1)
            # conv2_val = int(((conv1_val - 3 + 1) - (3 - 1) - 1) / 3 + 1)
            # lay_size = conv2_val * 64
            # # model = CNNModel(1, 32, 64, 3, lay_size, 128).to(device)
            conv1_size = 32
            conv2_size = 64
            kernel_size = 5
            node_size = 128

            # 定义神经网络模型
            # cnnModel = Sequential()
            # cnnModel.add(Conv1D(conv1_size, kernel_size=kernel_size, activation='relu', input_shape=(len(featureVectList[0]), 1)))
            # cnnModel.add(MaxPooling1D(pool_size=kernel_size))
            # cnnModel.add(Conv1D(conv2_size, kernel_size=kernel_size, activation='relu'))
            # cnnModel.add(MaxPooling1D(pool_size=kernel_size))
            # cnnModel.add(Flatten())
            # cnnModel.add(Dense(node_size, activation='linear'))
            # cnnModel.add(Dense(node_size, activation='linear'))
            # cnnModel.add(Dense(2, activation='linear'))

            # 编译模型
            cnnModel.compile(optimizer='adam',
                             loss='binary_crossentropy',
                             metrics=['accuracy'])

            cnnModel.fit(X_train_std, y_train, epochs=1, batch_size=64)
            if saveFile:
                saveModel(cnnModel, os.path.join(
                    SOMfaultlocalizationResultPath, project, version, "Model", mode + "_cnnModel"))
        # 用分类器预测测试集的结果
        y_pred = cnnModel.predict(X_test_std)

        # 将指标存储为字典
        test_loss, test_acc = cnnModel.evaluate(X_test, y_test)

        # save the model metrics
        results = {
            "accuracy": test_acc
        }

        checkAndCreateDir(os.path.join(
            SOMfaultlocalizationResultPath, project, version, "ModelMetrics"))

        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "ModelMetrics", mode + "_cnnModel.json"),"w") as f:
            f.write(json.dumps(results, indent=2))

        test_results = {indx_test[i]: y_pred[i] for i in range(len(indx_test))}
        result = np.empty(len(featureVectList), dtype=y_train.dtype)
        result[indx_train] = y_train
        result[indx_test] = [test_results.get(i, 0) for i in indx_test]

        return result

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None


# endregion


def saveModel(rf_clf, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(rf_clf, f)


def loadModel(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def executeModel(modelName, featureVectList, passLabelList, killLabelList, museKillLabelList, pkMap1, pkMap2, faultFileName1, faultFileName2,
                 testNum, saveFile, project, version, kill_data, i):
    try:
        # time1 = datetime.datetime.now()
        killResult = modelName(featureVectList, killLabelList, saveFile, project, version, "kill", kill_data, i)
        # time1 = datetime.datetime.now() - time1
        # logging.info(str(modelName).split(" ")[1])
        # logging.info(time1)
        susResult = calMbfl(project, version, passLabelList, killResult, pkMap1, pkMap2, faultFileName1, faultFileName2, testNum, 
                            f"pred_susFunc_pmbfl_{i}_2.json", i)
        topNResultBest,topNResultAverage,topNResultWorst = calTopNMbfl(project, version, susResult, f"pred_topN_pmbfl_{i}_2.json")

        MFR,MAR = calMFRandMAR(project, version, topNResultWorst, f"pred_TMP_pmbfl_{i}_2.json")

        print(project, version, str(modelName).split(" ")[1], "success")
        logging.info(f"{project} {version} {str(modelName).split(' ')[1]} success")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


def calMbfl(project, version, passList, killList, pkMap1, pkMap2, faultFileName1, faultFileName2, testNum, fileName, i):
    '''
    通过变异体的执行矩阵和杀死矩阵计算语句怀疑度
    '''
    try:
        muInfoPath = f"{faultlocalizationResultPath}/{project}/{version}/muInfo.json"
        muResultPath = f"{faultlocalizationResultPath}/{project}/{version}/muResult.json"
        with open(muInfoPath, "r") as f:
            muInfo = json.load(f)
        with open(muResultPath, "r") as f:
            muResult = json.load(f)
        muInfo = random.sample(muInfo, round(len(killList) / testNum  * 0.01 * i))
        new_muResult = []
        for item in muInfo:
            for value in muResult:
                if value['index'] == item['index']:
                    new_muResult.append(value)
                    break
        muResult = new_muResult
        susResult = {}
        fomResult = {}
        sumResult = {}
        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project))
        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project, version))
        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project, version, 'predSuspicious'))

        suspiciousFirstOrderPath = os.path.join(SOMfaultlocalizationResultPath, project, version, 'predSuspicious', fileName)
        susResult = dict()
        susFunc = dict()

        #region 计算语句级怀疑度
        # for i in range(0, len(passList) // testNum):
        for i in range(0, len(faultFileName1)):
            if susResult.get(faultFileName1[i]) == None:
                susResult[faultFileName1[i]] = dict()
            Anp = 0
            Anf = 0
            Akp = 0
            Akf = 0
            for index in range(0, testNum):
                if passList[index + i * testNum] == 1:
                    if killList[index + i * testNum] == 1:
                        Akf += 1
                    else:
                        Anf += 1
                else:
                    if killList[index + i * testNum] == 1:
                        Akp += 1
                    else:
                        Anp += 1
            for method in mbflMethods:
                if susResult[faultFileName1[i]].get(str(method).split(" ")[1]) == None:
                    susResult[faultFileName1[i]][str(method).split(" ")[1]] = dict()
                if susResult[faultFileName1[i]][str(method).split(" ")[1]].get(pkMap1[i]) == None:
                    susResult[faultFileName1[i]][str(method).split(" ")[1]][pkMap1[i]] = method(Akf, Anf, Akp, Anp)
                else:
                    susResult[faultFileName1[i]][str(method).split(" ")[1]][pkMap1[i]] = max(
                        susResult[faultFileName1[i]][str(method).split(" ")[1]][pkMap1[i]], method(Akf, Anf, Akp, Anp))
                
                if susResult[faultFileName1[i]][str(method).split(" ")[1]].get(pkMap2[i]) == None:
                    susResult[faultFileName1[i]][str(method).split(" ")[1]][pkMap2[i]] = method(Akf, Anf, Akp, Anp)
                else:
                    susResult[faultFileName1[i]][str(method).split(" ")[1]][pkMap2[i]] = max(
                        susResult[faultFileName1[i]][str(method).split(" ")[1]][pkMap2[i]], method(Akf, Anf, Akp, Anp))
                    
        for i in range(0, len(faultFileName2)):
            if susResult.get(faultFileName2[i]) == None:
                susResult[faultFileName2[i]] = dict()
            Anp = 0
            Anf = 0
            Akp = 0
            Akf = 0
            for index in range(0, testNum):
                if passList[index + i * testNum] == 1:
                    if killList[index + i * testNum] == 1:
                        Akf += 1
                    else:
                        Anf += 1
                else:
                    if killList[index + i * testNum] == 1:
                        Akp += 1
                    else:
                        Anp += 1
            for method in mbflMethods:
                if susResult[faultFileName2[i]].get(str(method).split(" ")[1]) == None:
                    susResult[faultFileName2[i]][str(method).split(" ")[1]] = dict()
                if susResult[faultFileName2[i]][str(method).split(" ")[1]].get(pkMap1[i]) == None:
                    susResult[faultFileName2[i]][str(method).split(" ")[1]][pkMap1[i]] = method(Akf, Anf, Akp, Anp)
                else:
                    susResult[faultFileName2[i]][str(method).split(" ")[1]][pkMap1[i]] = max(
                        susResult[faultFileName2[i]][str(method).split(" ")[1]][pkMap1[i]], method(Akf, Anf, Akp, Anp))
                
                if susResult[faultFileName2[i]][str(method).split(" ")[1]].get(pkMap2[i]) == None:
                    susResult[faultFileName2[i]][str(method).split(" ")[1]][pkMap2[i]] = method(Akf, Anf, Akp, Anp)
                else:
                    susResult[faultFileName2[i]][str(method).split(" ")[1]][pkMap2[i]] = max(
                        susResult[faultFileName2[i]][str(method).split(" ")[1]][pkMap2[i]], method(Akf, Anf, Akp, Anp))
        
        
        for i in range(0, len(muResult)):
            if muResult[i]["status"] == 0:
                continue
            Anp = 0
            Anf = 0
            Akp = 0
            Akf = 0
            if fomResult.get(muInfo[i]["relativePath"]) == None:
                fomResult[muInfo[i]["relativePath"]] = dict()
            for index in range(0, len(muResult[i]["passList"][f"type3"])):
                if muResult[i]["passList"][f"type3"][index] == 1:
                    if muResult[i]["killList"][f"type3"][index] == 1:
                        Akf += 1
                    else:
                        Anf += 1
                else:
                    if muResult[i]["killList"][f"type3"][index] == 1:
                        Akp += 1
                    else:
                        Anp += 1
            for method in mbflMethods:
                if (fomResult[muInfo[i]["relativePath"]].get(str(method).split(" ")[1])== None):
                    fomResult[muInfo[i]["relativePath"]][str(method).split(" ")[1]] = dict()
                if (fomResult[muInfo[i]["relativePath"]][str(method).split(" ")[1]].get(muResult[i]["linenum"])== None):
                    fomResult[muInfo[i]["relativePath"]][str(method).split(" ")[1]][muResult[i]["linenum"]] = method(Akf, Anf, Akp, Anp)
                else:
                    fomResult[muInfo[i]["relativePath"]][str(method).split(" ")[1]][muResult[i]["linenum"]] = max(method(Akf, Anf, Akp, Anp), fomResult[muInfo[i]["relativePath"]][str(method).split(" ")[1]][muResult[i]["linenum"]])
        for item in susResult.keys():
            for method in mbflMethods:
                if sumResult.get(item) is None:
                    sumResult[item] = dict()
                if sumResult[item].get(str(method).split(" ")[1]) is None:
                    sumResult[item][str(method).split(" ")[1]] = dict()
                for line_num in susResult[item][str(method).split(" ")[1]]:
                    if sumResult[item][str(method).split(" ")[1]].get(line_num) is None:
                        sumResult[item][str(method).split(" ")[1]][line_num] = 0
                    sumResult[item][str(method).split(" ")[1]][line_num] += susResult[item][str(method).split(" ")[1]][line_num]
        for item in fomResult.keys():
            for method in mbflMethods:
                if sumResult.get(item) is None:
                    sumResult[item] = dict()
                if sumResult[item].get(str(method).split(" ")[1]) is None:
                    sumResult[item][str(method).split(" ")[1]] = dict()
                for line_num in fomResult[item][str(method).split(" ")[1]]:
                    if sumResult[item][str(method).split(" ")[1]].get(line_num) is None:
                        sumResult[item][str(method).split(" ")[1]][line_num] = 0
                    sumResult[item][str(method).split(" ")[1]][line_num] += fomResult[item][str(method).split(" ")[1]][line_num]

        
        for item in sumResult.keys():
            for method in mbflMethods:
                sumResult[item][str(method).split(" ")[1]] = dict(
                    sorted(sumResult[item][str(method).split(" ")[1]].items(),
                           key=operator.itemgetter(1), reverse=True))
        with open(suspiciousFirstOrderPath.replace('susFunc','susState'), 'w') as f:
            f.write(json.dumps(sumResult, indent=2))
        #endregion

        #region 计算方法级怀疑度
        with open("../../d4j/hugeToFunction/" + project + "/" + version + "/HugetoFunction.in", 'rb') as f:
            hugeToFunction = pickle.load(f)
        with open("../../d4j/outputClean/" + project + "/" + version + "/FunctionList.txt", 'r') as f:
            FunctionList = f.readlines()
        with open("../../d4j/outputClean/" + project + "/" + version + "/HugeToFile.txt", 'r') as f:
            hugeToFile = f.readlines()

        # 形成语句到函数的映射
        hugeToFiledict = dict()
        for i in range(0, len(hugeToFile)):
            if hugeToFiledict.get(hugeToFile[i].split("\t")[0]) == None:
                hugeToFiledict[hugeToFile[i].split("\t")[0]] = dict()
            functionLine = hugeToFunction[i] + 1
            count = sum(1 for element in FunctionList[0:functionLine] if
                        FunctionList[functionLine - 1].split(":")[0] in element)
            hugeToFiledict[hugeToFile[i].split("\t")[0]][hugeToFile[i].split("\t")[1].strip()] = count
        for key in sumResult.keys():
            susFunc[key] = dict()
            for method in sumResult[key].keys():
                susFunc[key][method] = dict()
                for line in sumResult[key][method].keys():
                    for k in hugeToFiledict.keys():
                        if k in key:
                            break
                    if hugeToFiledict[k].get(str(int(line) - 1)) == None:
                        continue
                    count = hugeToFiledict[k][str(int(line) - 1)]
                    if susFunc[key][method].get(count) == None:# 相当于取最大值
                        susFunc[key][method][count] = sumResult[key][method][line]

        with open(suspiciousFirstOrderPath, 'w') as f:
            f.write(json.dumps(susFunc, indent=2))
        #endregion
        print('finish pre mbfl')
        return susFunc
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')

def calTopNMbfl(project, version, susResult, FileName):
    try:
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, 'falutFunction.json'), 'r') as f:
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

        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project, version, "predTopNFunctionBest"))
        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project, version, "predTopNFunctionAverage"))
        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project, version, "predTopNFunctionWorst"))
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "predTopNFunctionBest", FileName), 'w') as f:
            f.write(json.dumps(topNResultBest, indent=2))
            f.close()
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "predTopNFunctionAverage", FileName),
                  'w') as f:
            f.write(json.dumps(topNResultAverage, indent=2))
            f.close()
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "predTopNFunctionWorst", FileName), 'w') as f:
            f.write(json.dumps(topNResultWorst, indent=2))
            f.close()

        return topNResultBest, topNResultAverage, topNResultWorst
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')

def calMFRandMAR(project, version, topNResult, FileName):
    try:
        MFR = dict()
        MAR = dict()
        cnt = 0
        for key in topNResult.keys():
            for line in topNResult[key].keys():
                cnt = cnt + 1
                for method, value in topNResult[key][line].items():
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

        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project, version, "predMFR"))
        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project, version, "predMAR"))
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "predMFR", FileName.replace('TMP','MFR')), 'w') as f:
            f.write(json.dumps(MFR, indent=2))
            f.close()
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "predMAR", FileName.replace('TMP','MAR')), 'w') as f:
            f.write(json.dumps(MAR, indent=2))
            f.close()

        return MFR, MAR
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')



def getMutantBasedSbfl():
    with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "subSOMInfo.json"), 'r') as f:
        muInfo = json.load(f)
    groups = {}
    for item in muInfo:
        linenum1 = item["linenum1"]
        linenum2 = item["linenum2"]
        
        if linenum1 not in groups:
            groups[linenum1] = []
        groups[linenum1].append(item)
        
        if linenum2 not in groups:
            groups[linenum2] = []
        groups[linenum2].append(item)
    
    
    new_groups = {}

    # 遍历每个组
    for key, values in groups.items():
        num_elements = max(1, round(len(values) * 2 / 10))
        selected_elements = random.sample(values, num_elements)
        new_groups[key] = selected_elements

    seen_indexes = set()
    for key, values in new_groups.items():
        for value in values:
            seen_indexes.add(value['index'])
    
    
    muInde = list()
    for i, item in enumerate(muInfo):
        if item["index"] in seen_indexes:
            muInde.append(i)

    return muInde

# 存储项目名称及其版本数
projectList = {
    "Chart": 26,
    "Cli": 39,
    "Closure": 176,
    "Codec": 18,
    "Collections": 4,
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
modelNames = [
    randomForestModel,
    # svmModel
    # ,
    # logisticRegressionModel,
    # knnModel,
    # naiveBayesModel,
    # mlpModel
    # , cnnModel
]
mbflMethods = [
    dstar
    # ,dstar_sub_one
    , 
    ochiai
    # ,ochiai_sub_one
    # ,ochiai_sub_two
    , gp13
    # # ,gp13_sub_one
    # # ,gp13_sub_two
    , op2
    # # ,op2_sub_one
    # # ,op2_sub_two
    , jaccard
    # # ,jaccard_sub_one
    , russell
    # # ,russell_sub_one
    , turantula
    # # ,turantula_sub_one
    , naish1
    , binary
    , crosstab
    # , dstar2
]
if __name__ == "__main__":
    # 打印当前使用的python环境路径
    print(sys.executable)
    logger = logger_config(log_path='PMBFL.log')
    pool = multiprocessing.Pool(processes=5)
    with open("./mutilFaultVersion.json", "r") as f:
        mutilFaultVersion = json.load(f)
    for project in projectList.keys():
        for versionNum in range(1, projectList[project] + 1):
            version = str(versionNum) + 'b'
            if mutilFaultVersion.get(project) is None or version not in mutilFaultVersion[project]:
                continue
            print(datetime.datetime.now())
            print(f"{project} {version} start")

            logging.info(f"{project} {version} start")
            featureVectList, passLabelList, killLabelList, museKillLabelList, pkMap1, pkMap2, testNum, faultFileName1, faultFileName2 = getTrainData(project, version)
            
            if featureVectList is None:
                continue
            # a = [3, 4, 5, 10, 15]
            a = [10]
            for i in a:
            # for i in range(1,2):
                muInde = getMutantBasedSbfl()

                kill_data = data_split(muInde, testNum, featureVectList, killLabelList, 0.01 * i)
                print(featureVectList.shape,passLabelList.shape,killLabelList.shape)
                print(len(museKillLabelList), len(pkMap1), len(faultFileName1))

                # 最好不要超过18
                while pool._taskqueue.qsize() > 3:
                    print(f"Waiting tasks in queue: {pool._taskqueue.qsize()}")
                    time.sleep(1 * 30)
                    

                saveFile = True
                for item in modelNames:
                    def err_call_back(err):
                        print(f'出错啦~ error:{str(err)}')

                    pool.apply_async(executeModel, (item, featureVectList, passLabelList,
                                                    killLabelList, museKillLabelList, pkMap1, pkMap2, faultFileName1, faultFileName2, testNum,
                                                    saveFile, project, version,kill_data, i), error_callback=err_call_back)
    print("all finish")

    logging.info("all finish")
    for thread in threadList:
        thread.join()
    pool.close()
    pool.join()
    exit(1)
