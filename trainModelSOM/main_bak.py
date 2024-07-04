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
    return 1 / math.sqrt(Akf + Akp)


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


# endregion


# 自动创建不存在的目录
def checkAndCreateDir(Path):
    if not os.path.exists(Path):
        os.mkdir(Path)


# 根据单个变异体的行号获取在hugeCode的行号，将行号转化为标准形式
def getHugeLine(project, version, info):
    try:
        hugeCodeLineInfo = info['mutFilePath'].split(
            f"{project}/{version}/{info['index']}/")[1]
        file_path = os.path.join(
            outputCleanPath, project, version, 'HugeToFile.txt')
        # Time 1b特殊处理，他的HugeToFile里面是  /time/Chronology.java	0 这种形式
        if project == 'Time' and version == '1b':
            hugeCodeLineInfo = '/time' + hugeCodeLineInfo.split('time')[1]

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
            for line_num, line in enumerate(lines, start=1):
                if line.strip() == f"{hugeCodeLineInfo}\t{info['linenum'] - 1}":
                    return line_num
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
        # 特征向量中加入变异算子的类型:
        operatorType = {'AOR': 0, 'LOR': 1, 'SOR': 2,
                        'COR': 3, 'ROR': 4, 'ORU': 5, 'LVR': 6, 'STD': 7}
        staticFeature.append(operatorType[info['typeOp']])
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
        line_num = getHugeLine(project, version, info)
        if not line_num:
            print(
                f"\033[1;31mMutant {info['index']} HugeToFile not fund\033[0m")
            logging.error(f"mMutant {info['index']} HugeToFile not fund")
            return False
        staticFeature = getStaticFeature(info, line_num, staticFea)
        # print(len(staticFeature))

        # 将list转化为numpy中的array，并转化成指定的形状
        staticFeature = np.array(staticFeature).reshape(-1, 28)
        dynamicFeature = np.array(dynamicFea).reshape(-1, 28)

        # 将staticFeature复制为(2199, 28)的二维数组
        staticFeature = np.tile(staticFeature, (dynamicFeature.shape[0], 1))
        # 将两个二维数组合并为(2199, 56)的二维数组
        featureVect = np.concatenate(
            (staticFeature, dynamicFeature), axis=1).tolist()
        # print(len(featureVect[0]))
        # 加入60个测试用例特征   24 * 3 = 72
        for i, item in enumerate(featureVect):
            for key in crFea.keys():
                if key == 'dstar2':
                    continue
                featureVect[i].append(crFea[key][i])
            # print(len(crFea))
            for key in sfFea.keys():
                if key == 'dstar2':
                    continue
                featureVect[i].append(sfFea[key][i])
            for key in ssFea.keys():
                if key == 'dstar2':
                    continue
                featureVect[i].append(ssFea[key][i])
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
        with open(os.path.join(faultlocalizationResultPath, project, version, "muInfo.json"), 'r') as f:
            muInfo = json.load(f)
        # muResult 的结果不正确，需要改正
        with open(os.path.join(faultlocalizationResultPath, project, version, "muResult.json"), 'r') as f:
            muResults = json.load(f)
        with open(os.path.join(allTestsPath, project, version, 'all_tests'), 'r') as f:
            allTests = [line.strip() for line in f.readlines()]
        with open(os.path.join(outputCleanPath, project, version, 'all_tests.txt'), 'r') as f:
            allTestsClean = [line.strip() for line in f.readlines()]
        for i, item in enumerate(allTestsClean):
            item = item.split('#')[1] + "(" + item.split('#')[0] + ")"
            if item in allTests:
                nodeleteLine.append(i)
        with open(os.path.join(djfeaturePath, project, version, "static_fea"), 'rb') as f:
            staticFea = pickle.load(f)
        with open(os.path.join(dynamicFeaturePath, project, version, "static_all"), 'rb') as f:
            dynamicFea = pickle.load(f)
            dynamicFea = [dynamicFea[i]
                          for i in range(len(dynamicFea)) if i in nodeleteLine]
        with open(os.path.join(tpydataPath, project, version, 'data_CR2_all'), 'rb') as f:
            crFea = pickle.load(f)
            for key in crFea.keys():
                crFea[key] = [crFea[key][i]
                              for i in range(len(crFea[key])) if i in nodeleteLine]
        with open(os.path.join(tpydataPath, project, version, 'data_SF_all'), 'rb') as f:
            sfFea = pickle.load(f)
            for key in sfFea.keys():
                sfFea[key] = [sfFea[key][i]
                              for i in range(len(sfFea[key])) if i in nodeleteLine]
        with open(os.path.join(tpydataPath, project, version, 'data_SS_all'), 'rb') as f:
            ssFea = pickle.load(f)
            for key in ssFea.keys():
                ssFea[key] = [ssFea[key][i]
                              for i in range(len(ssFea[key])) if i in nodeleteLine]

        nodeleteLine = list()
        for i, item in enumerate(allTests):
            item = item.split('(')[1].split(')')[0] + "#" + item.split('(')[0]
            if item in allTestsClean:
                nodeleteLine.append(i)
        return muInfo, muResults, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None, None, None, None, None, None, None, None


def process_muResults(muResults, project, version, muInfo, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine,error_flag):
    featureVectList = []
    passLabelList = []
    killLabelList = []
    museKillLabelList = []
    pkMap = []
    faultFileName = []

    for muResult in muResults:
        if error_flag.value:
            break
        i = muResult['index'] - 1
        info = muResult
        # 如果passList和killList为空那么就是执行失败了  应该是在生成的时候解决一下
        if all(not info['passList']['type{}'.format(i + 1)] for i in range(4)) and all(not info['killList']['type{}'.format(i + 1)] for i in range(4)):
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
                    print(f"\033[1;31m变异体{i}执行结果有误！\033[0m")
                    logging.error(f"变异体{i}执行结果有误！")
                    error_flag.value = True
                    break
                else:
                    featureVectList.extend(featureVect)  # [1805,128] -> [3610,128] 改为 [[1805,128],[1805,128]]
                    pkMap.append(info['linenum'])
                    faultFileName.append(muInfo[i]['relativePath'])
                    # pass标签可能用不到
                    passLabelList.extend([info['passList']['type{}'.format(i + 1)] for i in range(
                        len(info['passList'])) if i in nodeleteLine])
                    killLabelList.extend([info['killList']['type{}'.format(i + 1)] for i in range(
                        len(info['killList'])) if i in nodeleteLine])
                    museKillLabelList.extend([info['mKillList'][i] for i in range(
                        len(info['mKillList'])) if i in nodeleteLine])
    return featureVectList, passLabelList, killLabelList, museKillLabelList, pkMap, faultFileName


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


# 获取训练集和测试集数据  把128个特征拼成128维度的向量
def getTrainData(project, version):
    try:
        if ip != '202.4.130.30' and remoteFileExists(os.path.join(project, version, "data.tar.gz")):
            print("111")
            # return None, None, None, None, None, None, None
            return loadRemoteFile(os.path.join(project, version, "data.tar.gz"))
        print("222")
        featureVectList = []
        passLabelList = []
        killLabelList = []
        museKillLabelList = []
        pkMap = []
        faultFileName = []
        muInfo, muResults, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine = loadFile(
            project, version)
        if len(muInfo) != len(muResults):
            logging.error(f"变异体信息数量{len(muInfo)}不等于变异体执行结果数量{len(muResults)}")
            return None, None, None, None, None, None, None
        testNum = len(muResults[0]['passList'])
        if len(muResults) > 50:
            num_workers = 18
            chunk_size = len(muResults) // (2 * num_workers)
            with multiprocessing.Manager() as manager:
                error_flag = manager.Value('i', False)
                with multiprocessing.Pool(num_workers) as pools:
                    results = []
                    for i in range(0, len(muResults), chunk_size):
                        chunk = muResults[i:i + chunk_size]
                        results.append(pools.apply_async(process_muResults, args=(
                            chunk, project, version, muInfo, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine,
                            error_flag)))
                    for result in results:
                        featureVect, passLabel, killLabel, museKillLabel, pk, faultFile = result.get()
                        featureVectList.extend(featureVect)
                        passLabelList.extend(passLabel)
                        killLabelList.extend(killLabel)
                        museKillLabelList.extend(museKillLabel)
                        pkMap.extend(pk)
                        faultFileName.extend(faultFile)
                    if error_flag.value:
                        return None, None, None, None, None, None, None

                    def saveRemoteFileAsync(data, remote_path):
                        thread = threading.Thread(target=saveRemoteFile, args=(data, remote_path))
                        thread.start()

                    # saveRemoteFileAsync(featureVectList, os.path.join(project, version, "data", "featureVectList"))
                    # saveRemoteFileAsync(passLabelList, os.path.join(project, version, "data", "passLabelList"))
                    # saveRemoteFileAsync(killLabelList, os.path.join(project, version, "data", "killLabelList"))
                    # saveRemoteFileAsync(museKillLabelList, os.path.join(project, version, "data", "museKillLabelList"))
                    # saveRemoteFileAsync(pkMap, os.path.join(project, version, "data", "pkMap"))
                    # saveRemoteFileAsync(testNum, os.path.join(project, version, "data", "testNum"))
                    # saveRemoteFileAsync(faultFileName, os.path.join(project, version, "data", "faultFileName"))
                    return np.array(featureVectList), np.array(passLabelList), np.array(killLabelList), np.array(
                        museKillLabelList), pkMap, testNum, faultFileName
        else:
            testNum = len(muResults[0]['passList'])
            for i, info in enumerate(muResults):
                if info['status'] == 0:
                    continue
                featureVect = getFeatureVect(
                    project, version, muInfo[i], info, staticFea, dynamicFea, crFea, sfFea, ssFea)
                if not type(featureVect) is list:
                    continue
                if len([info['passList'][i] for i in range(len(info['passList'])) if i in nodeleteLine]) != len(
                        featureVect):
                    print(f"\033[1;31m变异体{i}执行结果有误！\033[0m")
                    logging.error(f"变异体{i}执行结果有误！")
                    return None, None, None, None, None, None, None
                featureVectList.extend(featureVect)
                pkMap.append(info['linenum'])
                faultFileName.append(muInfo[i]['relativePath'])
                passLabelList.extend([info['passList'][i] for i in range(
                    len(info['passList'])) if i in nodeleteLine])
                killLabelList.extend([info['killList'][i] for i in range(
                    len(info['killList'])) if i in nodeleteLine])
                museKillLabelList.extend([info['mKillList'][i] for i in range(
                    len(info['mKillList'])) if i in nodeleteLine])

            def saveRemoteFileAsync(data, remote_path):
                thread = threading.Thread(target=saveRemoteFile, args=(data, remote_path))
                thread.start()

            # saveRemoteFileAsync(featureVectList, os.path.join(project, version, "data", "featureVectList"))
            # saveRemoteFileAsync(passLabelList, os.path.join(project, version, "data", "passLabelList"))
            # saveRemoteFileAsync(killLabelList, os.path.join(project, version, "data", "killLabelList"))
            # saveRemoteFileAsync(museKillLabelList, os.path.join(project, version, "data", "museKillLabelList"))
            # saveRemoteFileAsync(pkMap, os.path.join(project, version, "data", "pkMap"))
            # saveRemoteFileAsync(testNum, os.path.join(project, version, "data", "testNum"))
            # saveRemoteFileAsync(faultFileName, os.path.join(project, version, "data", "faultFileName"))
            return np.array(featureVectList), np.array(passLabelList), np.array(killLabelList), np.array(
                museKillLabelList), pkMap, testNum, faultFileName
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None, None, None, None, None, None, None


# region 模型
def randomForestModel(featureVectList, labelList, saveFile, project, version, mode):
    try:
        indx = np.arange(len(featureVectList))
        # 分离训练和测试集
        X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(
            featureVectList, labelList, indx, test_size=0.9, shuffle=True)

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_rfModel")):
            rf_clf = loadModel(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_rfModel"))
        else:
            # 定义随机森林分类器
            rf_clf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1)
            rf_clf.fit(X_train, y_train)

            if saveFile:
                saveModel(rf_clf, os.path.join(
                    faultlocalizationResultPath, project, version, "Model", mode + "_rfModel"))

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

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(faultlocalizationResultPath, project, version, "ModelMetrics", mode + "_rfModel.json"),
                  "w") as f:
            json.dump(results, f)

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


def svmModel(featureVectList, labelList, saveFile, project, version, mode):
    try:
        indx = np.arange(len(featureVectList))
        # 分离训练和测试集
        X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(
            featureVectList, labelList, indx, test_size=0.9, shuffle=True)
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_svmModel")):
            svm_clf = loadModel(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_svmModel"))
        else:
            n_estimators = 10
            svm_clf = SVC(kernel='linear', C=0.1, cache_size=200, shrinking=True)
            clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'),
                                                        max_samples=1.0 / n_estimators, n_estimators=n_estimators))
            svm_clf.fit(X_train, y_train)

            if saveFile:
                saveModel(svm_clf, os.path.join(
                    faultlocalizationResultPath, project, version, "Model", mode + "_svmModel"))

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
            faultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(faultlocalizationResultPath, project, version, "ModelMetrics", mode + "svmModel.json"),
                  "w") as f:
            json.dump(results, f)

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


def logisticRegressionModel(featureVectList, labelList, saveFile, project, version, mode):
    try:
        indx = np.arange(len(featureVectList))
        # 分离训练和测试集
        X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(
            featureVectList, labelList, indx, test_size=0.9, shuffle=True)

        # 标准化特征向量
        scaler = Normalizer().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_lrModel")):
            lr_clf = loadModel(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_lrModel"))
        else:
            # 定义逻辑回归分类器
            lr_clf = LogisticRegression(solver='liblinear')
            lr_clf.fit(X_train_std, y_train)

            if saveFile:
                saveModel(lr_clf, os.path.join(
                    faultlocalizationResultPath, project, version, "Model", mode + "_lrModel"))

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
            faultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(faultlocalizationResultPath, project, version, "ModelMetrics", mode + "lrModel.json"),
                  "w") as f:
            json.dump(results, f)

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


def knnModel(featureVectList, labelList, saveFile, project, version, mode):
    try:
        indx = np.arange(len(featureVectList))
        # 分离训练和测试集
        X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(
            featureVectList, labelList, indx, test_size=0.9, shuffle=True)
        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_knnModel")):
            knn_clf = loadModel(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_knnModel"))
        else:
            knn_clf = KNeighborsClassifier(n_neighbors=3, leaf_size=10, weights='distance', algorithm='kd_tree',
                                           n_jobs=-1)
            knn_clf.fit(X_train_std, y_train)

            if saveFile:
                saveModel(knn_clf, os.path.join(
                    faultlocalizationResultPath, project, version, "Model", mode + "_knnModel"))

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
            faultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(faultlocalizationResultPath, project, version, "ModelMetrics", mode + "knnModel.json"),
                  "w") as f:
            json.dump(results, f)

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


def naiveBayesModel(featureVectList, labelList, saveFile, project, version, mode):
    try:
        indx = np.arange(len(featureVectList))
        # 分离训练和测试集
        X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(
            featureVectList, labelList, indx, test_size=0.9, shuffle=True)
        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_nbModel")):
            nb_clf = loadModel(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_nbModel"))
        else:
            nb_clf = GaussianNB()
            nb_clf.fit(X_train_std, y_train)

            if saveFile:
                saveModel(nb_clf, os.path.join(
                    faultlocalizationResultPath, project, version, "Model", mode + "_nbModel"))

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
            faultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(faultlocalizationResultPath, project, version, "ModelMetrics", mode + "nbModel.json"),
                  "w") as f:
            json.dump(results, f)

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


def mlpModel(featureVectList, labelList, saveFile, project, version, mode):
    try:
        indx = np.arange(len(featureVectList))
        # 分离训练和测试集
        X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(
            featureVectList, labelList, indx, test_size=0.9, shuffle=True)
        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_mlpModel")):
            mlp_clf = loadModel(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_mlpModel"))
        else:
            mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10000)
            mlp_clf.fit(X_train_std, y_train)

            if saveFile:
                saveModel(mlp_clf, os.path.join(
                    faultlocalizationResultPath, project, version, "Model", mode + "_mlpModel"))

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
            faultlocalizationResultPath, project, version, "ModelMetrics"))

        # 将字典写入json文件
        with open(os.path.join(faultlocalizationResultPath, project, version, "ModelMetrics", mode + "mlpModel.json"),
                  "w") as f:
            json.dump(results, f)

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


def cnnModel(featureVectList, labelList, saveFile, project, version, mode):
    try:
        indx = np.arange(len(featureVectList))
        # 分离训练和测试集
        X_train, X_test, y_train, y_test, indx_train, indx_test = train_test_split(
            featureVectList, labelList, indx, test_size=0.9, shuffle=True)
        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.fit_transform(X_test)
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "Model"))
        if os.path.isfile(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_cnnModel")):
            mlp_clf = loadModel(os.path.join(
                faultlocalizationResultPath, project, version, "Model", mode + "_cnnModel"))
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
                    faultlocalizationResultPath, project, version, "Model", mode + "_cnnModel"))
        # 用分类器预测测试集的结果
        y_pred = cnnModel.predict(X_test_std)

        # 将指标存储为字典
        test_loss, test_acc = cnnModel.evaluate(X_test, y_test)

        # save the model metrics
        results = {
            "accuracy": test_acc
        }

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "ModelMetrics"))

        with open(os.path.join(faultlocalizationResultPath, project, version, "ModelMetrics", mode + "_cnnModel.json"),
                  "w") as f:
            json.dump(results, f)

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


def calMbfl(project, version, passList, killList, pkMap, faultFileName, testNum, fileName):
    '''
    通过变异体的执行矩阵和杀死矩阵计算语句怀疑度
    '''
    try:
        suspiciousFirstOrderPath = os.path.join(
            faultlocalizationResultPath, project, version, 'suspicious', fileName)
        susResult = dict()

        for i in range(0, len(passList) // testNum):
            if susResult.get(faultFileName[i]) == None:
                susResult[faultFileName[i]] = dict()
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
                if susResult[faultFileName[i]].get(str(method).split(" ")[1]) == None:
                    susResult[faultFileName[i]][str(
                        method).split(" ")[1]] = dict()
                if susResult[faultFileName[i]][str(method).split(" ")[1]].get(pkMap[i]) == None:
                    susResult[faultFileName[i]][str(method).split(
                        " ")[1]][pkMap[i]] = method(Akf, Anf, Akp, Anp)
                else:
                    susResult[faultFileName[i]][str(method).split(" ")[1]][pkMap[i]] = max(
                        susResult[faultFileName[i]][str(method).split(" ")[1]][pkMap[i]], method(Akf, Anf, Akp, Anp))
        for item in susResult.keys():
            for method in mbflMethods:
                susResult[item][str(method).split(" ")[1]] = dict(
                    sorted(susResult[item][str(method).split(" ")[1]].items(),
                           key=operator.itemgetter(1), reverse=True))
        checkAndCreateDir(os.path.join(faultlocalizationResultPath, project))
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version))
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, 'suspicious'))
        with open(suspiciousFirstOrderPath, 'w') as f:
            f.write(json.dumps(susResult, indent=2))
        #     为什么为空
        return susResult
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


def calc_sus(i, passList, museKillList, testNum):
    # |fP|
    x1 = np.sum(np.logical_xor(passList[i * testNum:(i + 1) * testNum], museKillList[i * testNum:(i + 1) * testNum]))
    # |fP (s) ∩ pm|
    x2 = np.sum(np.logical_and(np.logical_not(passList[i * testNum:(i + 1) * testNum]),
                               museKillList[i * testNum:(i + 1) * testNum]))
    # |pP (s) ∩ fm|
    x3 = np.sum(np.logical_and(passList[i * testNum:(i + 1) * testNum], museKillList[i * testNum:(i + 1) * testNum]))
    # |pP|
    x4 = np.sum(np.logical_not(
        np.logical_xor(passList[i * testNum:(i + 1) * testNum], museKillList[i * testNum:(i + 1) * testNum])))
    # |mut(P )|·|fP|
    y1 = len(passList) // testNum * x1
    # |mut(P )|·|pP|
    y2 = len(passList) // testNum * x3
    f2p = x2
    p2f = x3
    if y1 == 0 or p2f == 0:
        alpha = sys.float_info.max
    else:
        alpha = f2p / y1 * y2 / p2f
    p = sys.float_info.max if x1 == 0 else x2 / x1
    q = sys.float_info.max if x4 == 0 else alpha * x3 / x4
    return p - q


def calMuse(project, version, passList, museKillList, pkMap, faultFileName, testNum, fileName):
    try:
        suspiciousFirstOrderPath = os.path.join(
            faultlocalizationResultPath, project, version, 'suspicious', fileName)
        susResult = dict()
        with ThreadPoolExecutor() as executor:
            futures = {}
            for i in range(0, len(passList) // testNum):
                fName = faultFileName[i]
                fLine = pkMap[i]
                if (fName, fLine) not in futures:
                    future = executor.submit(calc_sus, i, passList, museKillList, testNum)
                    futures[(fName, fLine)] = [future]
                else:
                    future = executor.submit(calc_sus, i, passList, museKillList, testNum)
                    futures[(fName, fLine)].append(future)
            for (fName, fLine), futures_list in futures.items():
                sus = sum([future.result() for future in futures_list])
                n = len(futures_list)
                if fName not in susResult:
                    susResult[fName] = {'muse': {}}
                susResult[fName]['muse'][fLine] = sus / n
        for item in susResult.keys():
            susResult[item]['muse'] = dict(sorted(susResult[item]['muse'].items(),
                                                  key=operator.itemgetter(1), reverse=True))
        checkAndCreateDir(os.path.join(faultlocalizationResultPath, project))
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version))
        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, 'suspicious'))
        with open(suspiciousFirstOrderPath, 'w') as f:
            f.write(json.dumps(susResult, indent=2))
        return susResult
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


def calTopNMbfl(project, version, susResult, FileName):
    try:
        with open(os.path.join(faultlocalizationResultPath, project, version, 'faultLocalization.json'), 'r') as f:
            faultLocalization = json.load(f)

        topNResult = dict()

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
                    firstNum = 0
                    lastNum = 0
                    t = -1
                    flag = True
                    for item, value in susResult[f][method].items():
                        lastNum += 1
                        if value != t:
                            firstNum = lastNum
                            t = value
                        if item == line:
                            topNResult[key][line][method] = firstNum
                            flag = False
                            break
                    if flag:
                        topNResult[key][line][method] = -1

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "topN"))
        with open(os.path.join(faultlocalizationResultPath, project, version, "topN", FileName), 'w') as f:
            f.write(json.dumps(topNResult, indent=2))

        f.close()
        return topNResult
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


def count_lines(filename):
    wc_output = subprocess.check_output(['wc', '-l', filename])
    line_count = int(wc_output.split()[0])
    return line_count


def calMFRMbfl(project, version, topNResult, FileName):
    try:
        faultFileLine = list()
        with open(os.path.join(faultlocalizationResultPath, project, version, 'faultLocalization.json'), 'r') as f:
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
            faultlocalizationResultPath, project, version, "MFR"))
        with open(os.path.join(faultlocalizationResultPath, project, version, "MFR", FileName), 'w') as f:
            f.write(json.dumps(MFRResult, indent=2))

        f.close()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


def calMARMbfl(project, version, topNResult, FileName):
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
            MARResult[key] = MARResult[key] / len

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "MAR"))
        with open(os.path.join(faultlocalizationResultPath, project, version, "MAR", FileName), 'w') as f:
            f.write(json.dumps(MARResult, indent=2))

        f.close()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


def calStatisticsSbfl(project, version, topNFileName, MFRFileName, MARFileName):
    try:
        topNResult = dict()

        with open(os.path.join(faultlocalizationResultPath, project, version, "suspiciousSbfl.json"), 'r') as f:
            suspiciousSbfl = json.load(f)

        with open(os.path.join(faultlocalizationResultPath, project, version, 'faultLocalization.json'), 'r') as f:
            faultLocalization = json.load(f)

        for key in faultLocalization.keys():
            topNResult[key] = dict()
            for line in faultLocalization[key]:
                topNResult[key][line] = dict()
                for method in suspiciousSbfl[key].keys():
                    if method in [str(mbflMethods[i]).split(" ")[1] for i in range(len(mbflMethods))]:
                        firstNum = 0
                        lastNum = 0
                        t = -1
                        flag = True
                        for item in suspiciousSbfl[key][method].keys():
                            lastNum += 1
                            if suspiciousSbfl[key][method][item] != t:
                                firstNum = lastNum
                                t = suspiciousSbfl[key][method][item]
                            if item == str(line):
                                topNResult[key][line][method] = firstNum
                                flag = False
                                break
                        if flag:
                            topNResult[key][line][method] = -1

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "topN"))
        with open(os.path.join(faultlocalizationResultPath, project, version, "topN", topNFileName), 'w') as f:
            f.write(json.dumps(topNResult, indent=2))

        faultFileLine = list()
        with open(os.path.join(faultlocalizationResultPath, project, version, 'faultLocalization.json'), 'r') as f:
            faultLocalization = json.load(f)
            for key in faultLocalization.keys():
                faultFileLine.append(count_lines(
                    os.path.join(djSrcPath, project, version, key[1:])))

        MFRResult = dict()
        for key in topNResult.keys():
            for line, methods in topNResult[key].items():
                for method, value in methods.items():
                    if MFRResult.get(method) == None:
                        MFRResult[method] = faultFileLine[0]
                    if value != -1 and MFRResult[method] < value:
                        MFRResult[method] = value

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "MFR"))
        with open(os.path.join(faultlocalizationResultPath, project, version, "MFR", MFRFileName), 'w') as f:
            f.write(json.dumps(MFRResult, indent=2))

        MARResult = dict()
        length = 0
        for key in topNResult.keys():
            for line, methods in topNResult[key].items():
                length += 1
                for method, value in methods.items():
                    if MARResult.get(method) == None:
                        MARResult[method] = 0
                    if value == -1:
                        MARResult[method] += count_lines(os.path.join(djSrcPath, project, version, key[1:]))
                    else:
                        MARResult[method] += value
        for key in MARResult.keys():
            MARResult[key] = MARResult[key] / length

        checkAndCreateDir(os.path.join(
            faultlocalizationResultPath, project, version, "MAR"))
        with open(os.path.join(faultlocalizationResultPath, project, version, "MAR", MARFileName), 'w') as f:
            f.write(json.dumps(MARResult, indent=2))

        f.close()

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


def executeModel(modelName, featureVectList, passLabelList, killLabelList, museKillLabelList, pkMap, faultFileName,testNum, saveFile, project, version):
    try:
        # 原程序的测试用例的pass 和 fail信息，直接读取然后搞成一致的格式就可以，不需要预测
        passResult = modelName(
            featureVectList, passLabelList, saveFile, project, version, "pass")
        # 把这个结果存成akf，akp,anp,anf四元组，直接在mutationTool那边计算怀疑度
        # killResult和killLabelList长的一模一样
        killResult = modelName(
            featureVectList, killLabelList, saveFile, project, version, "kill")
        # muse相关暂时不用
        musekillResult = modelName(
            featureVectList, museKillLabelList, saveFile, project, version, "musekill"
        )
        if isinstance(passResult, np.ndarray) and isinstance(killResult, np.ndarray):
            susResult = calMbfl(project, version, passResult, killResult, pkMap, faultFileName, testNum,
                                "pred_suspicious_first_order_" +
                                str(modelName).split(" ")[1] + ".json")
            topNResult = calTopNMbfl(
                project, version, susResult, "pred_topN_" + str(modelName).split(" ")[1] + ".json")
            calMFRMbfl(project, version, topNResult, "pred_MFR_" +
                       str(modelName).split(" ")[1] + ".json")
            calMARMbfl(project, version, topNResult, "pred_MAR_" +
                       str(modelName).split(" ")[1] + ".json")

            # muse相关暂时不用
            susResult = calMuse(project, version, passResult, musekillResult, pkMap, faultFileName, testNum,
                                "pred_suspicious_first_order_" +
                                str(modelName).split(" ")[1] + "_muse.json")
            topNResult = calTopNMbfl(
                project, version, susResult, "pred_topN_" + str(modelName).split(" ")[1] + "_muse.json")
            calMFRMbfl(project, version, topNResult, "pred_MFR_" +
                       str(modelName).split(" ")[1] + "_muse.json")
            calMARMbfl(project, version, topNResult, "pred_MAR_" +
                       str(modelName).split(" ")[1] + "_muse.json")
        print(project, version, str(modelName).split(" ")[1], "success")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')


# 存储项目名称及其版本数
projectList = {
    # "Chart": 26,
    # "Cli": 39,
    # "Closure": 176,
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
    # "Lang": 65,
    # "Math": 106,
    # "Mockito": 38,
    "Time": 27,
}
modelNames = [
    randomForestModel
    ,
    # svmModel
    # , 
    logisticRegressionModel
    ,
    knnModel
    , naiveBayesModel
    , mlpModel
    # , cnnModel
]
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

    logger = logger_config(log_path='main.log')
    pool = multiprocessing.Pool(processes=4)
    with open("../mutationTool/failVersion.json", "r") as f:
        failVersion = json.load(f)
    for project in projectList.keys():
        # project = "JacksonCore"
        # failVersion[project] = list()
        for versionNum in range(1, projectList[project] + 1):
            version = str(versionNum) + 'b'
            # version = "4b"
            if not failVersion.get(project) is None and version in failVersion[project]:
                continue
            print(datetime.datetime.now())
            print(f"{project} {version} start")

            logging.info(f"{project} {version} start")
            featureVectList, passLabelList, killLabelList, museKillLabelList, pkMap, testNum, faultFileName = getTrainData(
                project, version)
            if featureVectList is None:
                continue
            # print(len(featureVectList), len(passLabelList), len(killLabelList), len(museKillLabelList), len(pkMap))

            # 最好不要超过18
            while pool._taskqueue.qsize() > 6:
                print(f"Waiting tasks in queue: {pool._taskqueue.qsize()}")
                time.sleep(3 * 60)

            # 计算传统mbfl的结果 ，目前是语句级
            susResult = calMbfl(project, version, passLabelList, killLabelList, pkMap, faultFileName,
                                testNum, "suspicious_first_order.json")
            topNResult = calTopNMbfl(
                project, version, susResult, 'topN_first_order.json')
            calMFRMbfl(project, version, topNResult, 'MFR_first_order.json')
            calMARMbfl(project, version, topNResult, 'MAR_first_order.json')
            # sbfl只是为了统计mar mfr那些，目前是语句级
            calStatisticsSbfl(project, version,
                              'topN_sbfl.json', 'MFR_sbfl.json', 'MAR_sbfl.json')

            # muse相关暂时不用
            susResult = calMuse(project, version, passLabelList, museKillLabelList, pkMap, faultFileName, testNum,
                                "suspicious_muse.json")
            topNResult = calTopNMbfl(
                project, version, susResult, 'topN_muse.json')
            calMFRMbfl(project, version, topNResult, 'MFR_muse.json')
            calMARMbfl(project, version, topNResult, 'MAR_muse.json')

            print(f"{project} {version} end")
            saveFile = True
            for item in modelNames:
                def err_call_back(err):
                    print(f'出错啦~ error:{str(err)}')


                pool.apply_async(executeModel, (item, featureVectList, passLabelList,
                                                killLabelList, museKillLabelList, pkMap, faultFileName, testNum,
                                                saveFile, project, version), error_callback=err_call_back)
    for thread in threadList:
        thread.join()
    pool.close()
    pool.join()
    exit(1)
