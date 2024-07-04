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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
import random
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
                if line.strip() == f"{hugeCodeLineInfo}\t{int(info['linenum']) - 1}":
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
            staticFeature.append(staticFea['fcci'][i][line_num-1])
        for item in staticFea.keys():
            if item == 'fcci':
                continue
            staticFeature.append(staticFea[item][line_num-1])
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

        # 将list转化为numpy中的array，并转化成指定的形状
        staticFeature = np.array(staticFeature).reshape(-1, 28)
        dynamicFeature = np.array(dynamicFea).reshape(-1, 28)

        # 将staticFeature复制为(2199, 28)的二维数组
        staticFeature = np.tile(staticFeature, (dynamicFeature.shape[0], 1))
        # 将两个二维数组合并为(2199, 56)的二维数组
        featureVect = np.concatenate(
            (staticFeature, dynamicFeature), axis=1).tolist()
        # 加入60个测试用例特征
        for i, item in enumerate(featureVect):
            for key in crFea.keys():
                if key == 'dstar2':
                    continue
                featureVect[i].append(crFea[key][i])
            for key in sfFea.keys():
                if key == 'dstar2':
                    continue
                featureVect[i].append(sfFea[key][i])
            for key in ssFea.keys():
                if key == 'dstar2':
                    continue
                featureVect[i].append(ssFea[key][i])
        return featureVect
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


def process_muResults(muResults, project, version, muInfo, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine,error_flag):
    featureVectList = []
    passLabelList = []
    killLabelList = []
    pkMap = []
    faultFileName = []


    for muResult in muResults:
        if error_flag.value:
            break
        i = int(muResult['index']) - 1
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
                    pkMap.append(info['linenum'])
                    faultFileName.append(muInfo[i]['relativePath'])
                    # 加4种type
                    # passLabelList.extend([info['passList']['type{}'.format(i + 1)] for i in range(
                    #     len(info['passList'])) if i in nodeleteLine])
                    # killLabelList.extend([info['killList']['type{}'.format(i + 1)] for i in range(
                    #     len(info['killList'])) if i in nodeleteLine])
                    # museKillLabelList.extend([info['mKillList'][i] for i in range(
                    #     len(info['mKillList'])) if i in nodeleteLine])

                    # 只加type4
                    passLabelList.extend(info['passList']['type4'])
                    killLabelList.extend(info['killList']['type4'])
    return featureVectList, passLabelList, killLabelList, pkMap, faultFileName

# 获取训练集和测试集数据
def getTrainData(project, version):
    try:
        featureVectList = []
        passLabelList = []
        killLabelList = []
        pkMap = []
        faultFileName = []
        muInfo, muResults, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine = loadFile(
            project, version)
        if len(muInfo) != len(muResults):
            print(f"{project} {version} 变异体信息数量{len(muInfo)}不等于变异体执行结果数量{len(muResults)}")
            logging.error(f"{project} {version} 变异体信息数量{len(muInfo)}不等于变异体执行结果数量{len(muResults)}")
            return None, None, None, None, None, None, None
             
        testNum = len(muResults[0]['passList']['type4'])

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
                    results.append(pools.apply_async(process_muResults, args=(
                        chunk, project, version, muInfo, staticFea, dynamicFea, crFea, sfFea, ssFea, nodeleteLine,
                        error_flag)))
                for result in results:
                    featureVect, passLabel, killLabel, museKillLabel, pk, faultFile = result.get()
                    featureVectList.extend(featureVect)
                    passLabelList.extend(passLabel)
                    killLabelList.extend(killLabel)
                    pkMap.extend(pk)
                    faultFileName.extend(faultFile)
                if error_flag.value:
                    return None, None, None, None, None, None
                
                return np.array(featureVectList), np.array(passLabelList), np.array(killLabelList), pkMap, testNum, faultFileName
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
        logging.error(f'Error at line {line_number}: {e}')
        return None, None, None, None, None, None


def calMbfl(project, version, passList, killList, pkMap, faultFileName, testNum, fileName):
    '''
    通过变异体的执行矩阵和杀死矩阵计算语句怀疑度
    '''
    try:
        suspiciousFirstOrderPath = os.path.join(
            faultlocalizationResultPath, project, version, 'suspicious', fileName)
        susResult = dict()

        for i in range(0, len(passList)//testNum):
            if susResult.get(faultFileName[i]) == None:
                susResult[faultFileName[i]] = dict()
            Anp = 0
            Anf = 0
            Akp = 0
            Akf = 0
            for index in range(0, testNum):
                if passList[index+i*testNum] == 1:
                    if killList[index+i*testNum] == 1:
                        Akf += 1
                    else:
                        Anf += 1
                else:
                    if killList[index+i*testNum] == 1:
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
                susResult[item][str(method).split(" ")[1]] = dict(sorted(susResult[item][str(method).split(" ")[1]].items(),
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
        with open(os.path.join(faultlocalizationResultPath, project, version, 'faultLocalization.json'), 'r')as f:
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
        with open(os.path.join(faultlocalizationResultPath, project, version, 'faultLocalization.json'), 'r')as f:
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
                        MARResult[method] += count_lines(
                            os.path.join(djSrcPath, project, version, key[1:]))
                    else:
                        MARResult[method] += value
        for key in MARResult.keys():
            MARResult[key] = MARResult[key]/len

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


def getMutantBasedHMER(project, version,ratio=0.1):
    # 获取变异体个数
    with open(os.path.join(faultlocalizationResultPath, project, version, "muInfo.json"), 'r') as f:
        muInfo = json.load(f)
    muNum = int(len(muInfo)* ratio)
    with open(os.path.join(faultlocalizationResultPath, project, version, "suspiciousSbfl.json"), 'r') as f:
        suspiciousSbfl = json.load(f)
    for key in suspiciousSbfl.keys():
        suspiciousSbfl[key] = suspiciousSbfl[key]["dstar"]
    totalSupicious = 0
    muInde = list()
    # 计算总的怀疑度分数
    for key in suspiciousSbfl.keys():
        for k in suspiciousSbfl[key]:
            totalSupicious += suspiciousSbfl[key][k]
    for key in suspiciousSbfl.keys():
        if len(muInde) > muNum:
            break
        for k in suspiciousSbfl[key]:
            if len(muInde) > muNum:
                break
            nowNum = int(muNum * suspiciousSbfl[key][k] / totalSupicious)
            newMuInfo = []
            for d in muInfo:
                if int(d.get("linenum")) == int(k):
                    newMuInfo.append(int(d.get("index")))
                if len(newMuInfo) > nowNum:
                    newMuInfo = random.sample(newMuInfo, math.ceil(nowNum))
                    break
            muInde.extend(newMuInfo)
    
    mutList = []
    infoList = []
    for item in muInfo:
        if int(item['index']) in muInde:
            infoList.append(item)
    with open(os.path.join(faultlocalizationResultPath, project, version, "muResult.json"), 'r') as f:
        muResult = json.load(f)
    for item in muResult:
        if int(item['index']) in muInde:
            mutList.append(item)
    print(f'hmer nums: {len(mutList)}')
    return infoList, mutList


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
                if file != 'hmer.json':
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
        logging.info(f"{project} {version} success")
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
            "hmer.json",
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


def HMER(project, version,rate=0.1):
    infoList, mutList = getMutantBasedHMER(project, version,rate)
    calFomMbfl(project, version, infoList, mutList)
    countFunctionSus(project, version)

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

    logger = logger_config(log_path='hmer.log')
    with open("./failVersion.json", "r") as f:
        failVersion = json.load(f)

    pool = multiprocessing.Pool(processes=13)

    for project in projectList.keys():
        for versionNum in range(1, projectList[project] + 1):
            version = str(versionNum) + 'b'
            if not failVersion.get(project) is None and version in failVersion[project]:
                continue
            # 跳出已经执行过的版本
            # mfrPath = os.path.join(
            #     faultlocalizationResultPath, project, version, "MFR")
            # if not os.path.exists(mfrPath + "/pred_MFR_randomForestModel_pmt.json"):
            #     continue
            # if os.path.exists(mfrPath + "/MFR_hmer.json"):
            #     continue

            logging.info(f"{project} {version} start")

            # 最好不要超过18
            while pool._taskqueue.qsize() > 9:
                print(f"Waiting tasks in queue: {pool._taskqueue.qsize()}")
                time.sleep(15)
            pool.apply_async(HMER, (project,version,0.3))

            # infoList, mutList = getMutantBasedHMER(project, version,0.1)
            # calFomMbfl(project, version, infoList, mutList)
            # countFunctionSus(project, version)
            


            # susResult = calMbfl(project, version, newpassLabelList, newkillLabelList, newmutantLineMap, faultFileName,
            #                     testNum, "suspicious_hmer.json")
            # topNResult = calTopNMbfl(
            #     project, version, susResult, 'topN_hmer.json')
            # calMFRMbfl(project, version, topNResult, 'MFR_hmer.json')
            # calMARMbfl(project, version, topNResult, 'MAR_hmer.json')
    pool.close()
    pool.join()
    logging.info("all finish")
    exit(1)
