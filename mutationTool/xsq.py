import json
import os
import pickle
import sys

# 设置项目和版本
import shutil

import numpy as np
import math

project = 'Time'
version = '3b'

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

with open('./config.json', 'r') as configFile:
    configData = json.load(configFile)
tempSrcPath = configData['tempSrcPath']
tpydataPath = configData['tpydataPath']
outputCleanPath = configData['outputCleanPath']
djSrcPath = configData['djSrcPath']
mutantsFilePath = configData['mutantsFilePath']
faliingTestOutputPath = configData['faliingTestOutputPath']
faultlocalizationResultPath = configData['faultlocalizationResultPath']
FOMprocessedData = configData['FOMprocessedData']
SOMfaultlocalizationResultPath = configData['SOMfaultlocalizationResultPath']

with open(os.path.join(faultlocalizationResultPath, 'Lang', '11b', "failing_tests/100"), 'r', encoding='utf-8') as f:
    lines = f.read()
lines = lines.split('---')
lines = [s.strip() for s in lines if s.strip()]
faileTests = {
    'type1': [],
    'type2': [],
    'type3': [],
    'type4': []
}
for s in lines:
    # print(s)
    testName = s.split('\n')[0]
    faileTests['type1'].append([testName, s.split('\n')[0]])
    faileTests['type2'].append([testName, s.split('\n')[0] + s.split('\n')[1].split(':')[0]])
    faileTests['type3'].append([testName, s.split('\n')[0] + s.split('\n')[1]])
    faileTests['type4'].append([testName, s])
# print(faileTests['type1'][0])
# print(faileTests['type2'][0])
print(faileTests['type3'][0])
# print(faileTests['type4'][0])


suspiciousPath = faultlocalizationResultPath + "/" + project + "/" + version + "/susStatement"
for root, dirs, files in os.walk(suspiciousPath):
    print(root, dirs, files)
    print(files)

nodeleteLine = list()
with open(outputCleanPath + "/" + project + "/" + version + "/CoverageMatrix_Function.in", 'rb') as f:
    CoverageMatrix_Function = pickle.load(f)
print('CoverageMatrix_Function ',len(CoverageMatrix_Function),len(CoverageMatrix_Function[0]))
allTestsPath = '/home/fanluxi/d4j/all_tests/'
with open(os.path.join(allTestsPath, project, version, 'all_tests'), 'r') as f:
    allTests = [line.strip() for line in f.readlines()]
with open(os.path.join(outputCleanPath, project, version, 'all_tests.txt'), 'r') as f:
    allTestsClean = [line.strip() for line in f.readlines()]
for i, item in enumerate(allTestsClean):
    item = item.split('#')[1] + "(" + item.split('#')[0] + ")"
    if item in allTests:
        nodeleteLine.append(i)
nodeleteLine = list(range(len(allTestsClean)))
print(len(nodeleteLine),nodeleteLine[0])

# 特征
# djfeaturePath = "/home/public/d4j/feature"
# with open(os.path.join(djfeaturePath, project, version, "static_fea"), 'rb') as f:
#     staticFea = pickle.load(f)
#     print('staticFea ',len(staticFea))
#
# dynamicFeaturePath = "/home/public/d4j/dynamicFeature"
# with open(os.path.join(dynamicFeaturePath, project, version, "static_all"), 'rb') as f:
#     dynamicFea = pickle.load(f)
#     dynamicFea = [dynamicFea[i] for i in range(len(dynamicFea)) if i in nodeleteLine]
#     print('dynamicFea ',len(dynamicFea),len(dynamicFea[0]))
#
# with open(os.path.join(tpydataPath, project, version, 'CR'), 'rb') as f:
#     crFea = pickle.load(f)
#     for key in crFea.keys():
#         crFea[key] = [crFea[key][i] for i in range(len(crFea[key])) if i in nodeleteLine]
#     print('crFea ',len(crFea), len(crFea['ochiai']))
# with open(os.path.join(tpydataPath, project, version, 'SF'), 'rb') as f:
#     sfFea = pickle.load(f)
#     for key in sfFea.keys():
#         sfFea[key] = [sfFea[key][i] for i in range(len(sfFea[key])) if i in nodeleteLine]
#     print('sfFea ',len(sfFea), len(sfFea['ochiai']))
# with open(os.path.join(tpydataPath, project, version, 'SS'), 'rb') as f:
#     ssFea = pickle.load(f)
#     for key in ssFea.keys():
#         ssFea[key] = [ssFea[key][i] for i in range(len(ssFea[key])) if i in nodeleteLine]
#     print('ssFea ',len(ssFea), len(ssFea['ochiai']))
# with open(os.path.join(faultlocalizationResultPath,project,version,'muResult.json'))as f:
#     js = json.load(f)
#     print(len(js[0]['killList']['type4']))
#     print(len(js[0]['passList']['type4']))


# 删除muResult.json为空的版本
# w_path = os.path.join(faultlocalizationResultPath, project)
# w_list = os.listdir(w_path)
# for w in w_list:
#     v_path = os.path.join(w_path, w, 'muResult.json')
#     with open(v_path, 'r') as f:
#         muRes = json.load(f)
#     if len(muRes) == 0:
#         shutil.rmtree(os.path.join(w_path,w))

# 删除模型和模型评价
path = os.path.join(faultlocalizationResultPath, project)
dirs = os.listdir(path)
for di in dirs:
    print(di)
    dModel = os.path.join(path,di,'Model')
    dModelMetrics = os.path.join(path,di,'ModelMetrics')
    dPredSuspicious = os.path.join(path,di,'predSuspicious')
    dPredTopNFunctionAverage = os.path.join(path,di,'predTopNFunctionAverage')
    dPredTopNFunctionBest = os.path.join(path,di,'predTopNFunctionBest')
    dPredTopNFunctionWorst = os.path.join(path,di,'predTopNFunctionWorst')

    dPredSuspicious_BasedSbfl = os.path.join(path,di,'predSuspicious_BasedSbfl')
    dPredTopNFunctionAverage_BasedSbfl = os.path.join(path,di,'predTopNFunctionAverage_BasedSbfl')
    dPredTopNFunctionBest_BasedSbfl = os.path.join(path,di,'predTopNFunctionBest_BasedSbfl')
    dPredTopNFunctionWorst_BasedSbfl = os.path.join(path,di,'predTopNFunctionWorst_BasedSbfl')
    dMAR = os.path.join(path,di,'MAR')
    dMFR = os.path.join(path,di,'MFR')
    if os.path.exists(dModel):
        shutil.rmtree(dModel)
        # for filename in os.listdir(dModel):
        #     if "sbfl" not in filename:
        #         file_path = os.path.join(dModel, filename)
        #         if os.path.isfile(file_path):
        #             os.remove(file_path)
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)
    if os.path.exists(dModelMetrics):
        shutil.rmtree(dModelMetrics)
        # for filename in os.listdir(dModelMetrics):
        #     if "sbfl" not in filename:
        #         file_path = os.path.join(dModelMetrics, filename)
        #         if os.path.isfile(file_path):
        #             os.remove(file_path)
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)
    if os.path.exists(dPredSuspicious):
        shutil.rmtree(dPredSuspicious)
    if os.path.exists(dPredTopNFunctionAverage):
        shutil.rmtree(dPredTopNFunctionAverage)
    if os.path.exists(dPredTopNFunctionBest):
        shutil.rmtree(dPredTopNFunctionBest)
    if os.path.exists(dPredTopNFunctionWorst):
        shutil.rmtree(dPredTopNFunctionWorst)

    if os.path.exists(dPredSuspicious_BasedSbfl):
        shutil.rmtree(dPredSuspicious_BasedSbfl)
    if os.path.exists(dPredTopNFunctionAverage_BasedSbfl):
        shutil.rmtree(dPredTopNFunctionAverage_BasedSbfl)
    if os.path.exists(dPredTopNFunctionBest_BasedSbfl):
        shutil.rmtree(dPredTopNFunctionBest_BasedSbfl)
    if os.path.exists(dPredTopNFunctionWorst_BasedSbfl):
        shutil.rmtree(dPredTopNFunctionWorst_BasedSbfl)
    # if os.path.exists(dMAR):
    #     shutil.rmtree(dMAR)
    # if os.path.exists(dMFR):
    #     shutil.rmtree(dMFR)

projj = 'Math'
with open("./failVersion.json", "r") as f:
    failVersion = json.load(f)
timeList = os.listdir(os.path.join(faultlocalizationResultPath,projj))
for item in timeList:
    if item in failVersion[projj] or not item.endswith("b"):
        continue
    pa = os.path.join(faultlocalizationResultPath,projj, item,'falutFunction.json')
    with open(pa, 'r') as f:
        js = json.load(f)
        if len(js) == 0:
            print(item)

# region 重写topn、mar、mfr

# a = 'src/main/java/org/joda/time/field/UnsupportedDurationField.java'
# b = 'ochiai'
# c = a+'<419>'+b
# print(c.split('<419>'))
# # region mbfl计算公式
#
# def dstar(Akf, Anf, Akp, Anp):
#     if (Akp + Anf) == 0:
#         return sys.float_info.max
#     return math.pow(Akf, 3) / (Akp + Anf)
#
#
# def dstar_sub_one(Akf, Anf, Akp, Anp):
#     if (Akp + Anf) == 0:
#         return sys.float_info.max
#     return 1 / (Akp + Anf)
#
#
# def ochiai(Akf, Anf, Akp, Anp):
#     if (Akf + Anf) * (Akf + Akp) == 0:
#         return 0
#     return Akf / math.sqrt((Akf + Anf) * (Akf + Akp))
#
#
# def ochiai_sub_one(Akf, Anf, Akp, Anp):
#     if (Akf + Anf) == 0:
#         return 0
#     return 1 / math.sqrt(Anf + Akf)
#
#
# def ochiai_sub_two(Akf, Anf, Akp, Anp):
#     if (Akf + Akp) == 0:
#         return 0
#     return 1 / math.sqrt(Akf+Akp)
#
#
# def ochiai_sub_three(Akf, Anf, Akp, Anp):
#     if (Akf + Anf) * (Akf + Akp) == 0:
#         return 0
#     return 1 / math.sqrt((Akf + Anf) * (Akf + Akp))
#
#
# def ochiai_sub_four(Akf, Anf, Akp, Anp):
#     if (Akf + Anf) == 0:
#         return 0
#     return Akf / math.sqrt(Anf + Akf)
#
#
# def ochiai_sub_five(Akf, Anf, Akp, Anp):
#     if (Akf + Akp) == 0:
#         return 0
#     return Akf / math.sqrt(Akf + Akp)
#
#
# def gp13(Akf, Anf, Akp, Anp):
#     if (2 * Akp + Akf) == 0:
#         return 0
#     return Akf + (Akf / (2 * Akp + Akf))
#
#
# def gp13_sub_one(Akf, Anf, Akp, Anp):
#     if (2 * Akp + Akf) == 0:
#         return 0
#     return 1 / (2 * Akp + Akf)
#
#
# def gp13_sub_two(Akf, Anf, Akp, Anp):
#     if (2 * Akp + Akf) == 0:
#         return 0
#     return Akf / (2 * Akp + Akf)
#
#
# def op2(Akf, Anf, Akp, Anp):
#     # XSQ
#     if ((Akp + Anp) + 1) == 0:
#         return 0
#     return Akf - (Akp / ((Akp + Anp) + 1))
#
#
# def op2_sub_one(Akf, Anf, Akp, Anp):
#     return Akp / ((Akp + Anf) + 1)
#
#
# def op2_sub_two(Akf, Anf, Akp, Anp):
#     return 1 / ((Akp + Anf) + 1)
#
#
# def jaccard(Akf, Anf, Akp, Anp):
#     if (Akf + Anf + Akp) == 0:
#         return 0
#     return Akf / (Anf + Akf + Akp)
#
#
# def jaccard_sub_one(Akf, Anf, Akp, Anp):
#     if (Akf + Anf + Akp) == 0:
#         return 0
#     return 1 / (Anf + Akf + Akp)
#
#
# def russell(Akf, Anf, Akp, Anp):
#     # XSQ
#     if  (Akp + Anp + Akf + Anf) == 0:
#         return 0
#     return Akf / (Akp + Anp + Akf + Anf)
#
#
# def russell_sub_one(Akf, Anf, Akp, Anp):
#     return 1 / (Akp + Anp + Akf + Anf)
#
#
# def turantula(Akf, Anf, Akp, Anp):
#     if (Akf + Akp) == 0 or (Akf + Anf) == 0:
#         return 0
#     if ((Akf + Anf) != 0) and ((Akp + Anp) == 0):
#         return 1
#     return (Akf / (Akf + Anf)) / ((Akf / (Akf + Anf)) + (Akp / (Akp + Anp)))
#
#
# def turantula_sub_one(Akf, Anf, Akp, Anp):
#     if (Akf + Akp) == 0 or (Akf + Anf) == 0:
#         return 0
#     if ((Akf + Anf) != 0) and ((Akp + Anp) == 0):
#         return 1
#     return 1 / ((Akf / (Anf + Akf)) + (Akp / (Anp + Akp)))
#
#
# def turantula_sub_two(Akf, Anf, Akp, Anp):
#     if (Akf + Akp) == 0 or (Akf + Anf) == 0:
#         return 0
#     if ((Akf + Anf) != 0) and ((Akp + Anp) == 0):
#         return 1
#     return Akf / (Anf + Akf)
#
#
# def turantula_sub_three(Akf, Anf, Akp, Anp):
#     if (Akf + Akp) == 0 or (Akf + Anf) == 0:
#         return 0
#     if ((Akf + Anf) != 0) and ((Akp + Anp) == 0):
#         return 1
#     return 1 / (Anf + Akf)
#
#
# def naish1(Akf, Anf, Akp, Anp):
#
#     return -1 if Anf > 0 else Anp
#
#
# def binary(Akf, Anf, Akp, Anp):
#     return 0 if Akf > 0 else 1
#
#
# def dstar2(Akf, Anf, Akp, Anp):
#     if (Akp + Anf) == 0:
#         return sys.float_info.max
#     return (math.pow(Akf, 2)) / (Akp + Anf)
#
#
# def crosstab(Akf, Akp, Anf, Anp):
#     tf = Akf + Anf
#     tp = Akp + Anp
#     N = tf + tp
#     Sw = 0
#     Ncw = Akf + Akp
#     Nuw = Anf + Anp
#     try:
#         Ecfw = Ncw * (tf / N)
#         Ecsw = Ncw * (tp / N)
#         Eufw = Nuw * (tf / N)
#         Eusw = Nuw * (tp / N)
#         X2w = pow(Akf - Ecfw, 2) / Ecfw + pow(Akp - Ecsw, 2) / Ecsw + pow(Anf - Eufw, 2) / Eufw + pow(
#             Anp - Eusw, 2) / Eusw
#         yw = (Akf / tf) / (Akp / tp)
#         if yw > 1:
#             Sw = X2w
#         elif yw < 1:
#             Sw = -X2w
#     except:
#         Sw = 0
#     return Sw
# # endregion
# mbflMethods = [
#     dstar
#     # ,dstar_sub_one
#     , ochiai
#     # ,ochiai_sub_one
#     # ,ochiai_sub_two
#     , gp13
#     # ,gp13_sub_one
#     # ,gp13_sub_two
#     , op2
#     # ,op2_sub_one
#     # ,op2_sub_two
#     , jaccard
#     # ,jaccard_sub_one
#     , russell
#     # ,russell_sub_one
#     , turantula
#     # ,turantula_sub_one
#     , naish1
#     , binary
#     , crosstab
#     , dstar2
# ]
#
# with open(os.path.join(faultlocalizationResultPath, 'Time', '3b', 'falutFunction.json'), 'r') as f:
#     faultLocalization = json.load(f)
# with open(os.path.join(faultlocalizationResultPath, 'Time', '3b', 'predSuspicious','pred_susFunc_randomForestModel.json'), 'r') as f:
#     susResult = json.load(f)
#
# # 将真实错误函数信息拼接起来
# faultFunc = list()
# tmp = '<419>'
# for item in faultLocalization:
#     for index in faultLocalization[item]:
#         faultFunc.append(item[1:] + tmp + str(index))
#
# topNResultBest = dict()
# topNResultAverage = dict()
# topNResultWorst = dict()
#
# for method in mbflMethods:
#     method_name = method.__name__
#     # 将同一公式下的所有函数进行统一排序生成newSusResult
#     newSusResult= dict()
#     for item in susResult:
#         for suskey,value in susResult[item][method_name].items():
#             k = item + tmp + suskey
#             newSusResult[k] = value
#     newSusResult = dict(sorted(newSusResult.items(), key=lambda item: item[1], reverse=True))
#
#     for faultKey in faultFunc:
#         key = '/' + faultKey.split('<419>')[0]
#         line = faultKey.split('<419>')[1]
#         # region 创建字典
#         if topNResultBest.get(key) is None:
#             topNResultBest[key] = dict()
#         if topNResultBest[key].get(line) is None:
#             topNResultBest[key][line] = dict()
#         if topNResultAverage.get(key) is None:
#             topNResultAverage[key] = dict()
#         if topNResultAverage[key].get(line) is None:
#             topNResultAverage[key][line] = dict()
#         if topNResultWorst.get(key) is None:
#             topNResultWorst[key] = dict()
#         if topNResultWorst[key].get(line) is None:
#             topNResultWorst[key][line] = dict()
#         # endregion
#
#         if newSusResult.get(faultKey) is None:
#             topNResultBest[key][line][method_name] = -1
#             topNResultAverage[key][line][method_name] = -1
#             topNResultWorst[key][line][method_name] = -1
#             continue
#
#         faultSus = newSusResult[faultKey]
#
#         startFlagIndex = -1
#         repeatFaultTime = 0
#         endFlagIndex = -1
#         ind = 0
#         for item, value in newSusResult.items():
#             ind += 1
#             if math.isnan(value):
#                 continue
#             if value > faultSus:
#                 continue
#             if value == faultSus:
#                 if startFlagIndex == -1:
#                     startFlagIndex = ind
#                 else:
#                     if item in faultFunc:
#                         repeatFaultTime += 1
#             else:
#                 endFlagIndex = ind - 1 - repeatFaultTime
#                 break
#         # 最好排名
#         topNResultBest[key][line][method_name] = startFlagIndex
#         # 平均排名
#         if endFlagIndex == -1:
#             endFlagIndex = ind
#         if startFlagIndex == -1 or endFlagIndex == -1:
#             topNResultAverage[key][line][method_name] = -1
#         else:
#             topNResultAverage[key][line][method_name] = (startFlagIndex + endFlagIndex) / 2
#         # 最坏排名
#         topNResultWorst[key][line][method_name] = endFlagIndex
# print(topNResultWorst)
#
# MFR = dict()
# MAR = dict()
# cnt = 0
# for key in topNResultWorst.keys():
#     for line in topNResultWorst[key].keys():
#         cnt = cnt + 1
#         for method,value in topNResultWorst[key][line].items():
#             if MFR.get(method) is None:
#                 MFR[method] = value
#             if MFR[method] > value:
#                 MFR[method] = value
#
#             if MAR.get(method) is None:
#                 MAR[method] = 0
#             MAR[method] += value
# for method in MAR.keys():
#     MAR[method] = MAR[method] / cnt
# print(MFR)
# print(cnt)
# print(MAR)

# endregion

def group_by_score(suspiciousSbfl):
    # 将字典按value排序
    sorted_suspiciousSbfl = sorted(suspiciousSbfl.items(), key=lambda x: sum(x[1].values()), reverse=True)

    result = {}
    for key, value in sorted_suspiciousSbfl:
        # 将字典中的value按降序排序
        sorted_values = sorted(value.items(), key=lambda x: x[1], reverse=True)
        # 将value按照相同的数值进行合并
        merged_values = []
        temp = []
        for i in range(len(sorted_values)):
            if i == 0:
                temp.append(sorted_values[i][0])
            elif sorted_values[i][1] == sorted_values[i-1][1]:
                temp.append(sorted_values[i][0])
            else:
                merged_values.append(temp)
                temp = [sorted_values[i][0]]
        merged_values.append(temp)
        # 将key和merged_values加入到result中
        result[key] = merged_values
    return result


def getMutantBasedSbfl():
    # 获取变异体个数
    with open(os.path.join(faultlocalizationResultPath, project, version, "muInfo.json"), 'r') as f:
        muInfo = json.load(f)
    muNum = len(muInfo)//10
    with open(os.path.join(faultlocalizationResultPath, project, version, "suspiciousSbfl.json"), 'r') as f:
        suspiciousSbfl = json.load(f)
    # 只取出来一个公式的怀疑度
    for key in suspiciousSbfl.keys():
        suspiciousSbfl[key] = suspiciousSbfl[key]["ochiai_sub_one"]
    # 按照怀疑度分数划分档次，相同的怀疑度是同一个level，被划分到同一个组
    result = group_by_score(suspiciousSbfl)
    nowNum = 0
    level = 0
    muInde = list()
    while (1):
        flag = True
        for key in result.keys():
            if level >= len(result[key]):
                continue
            for item in muInfo:
                if item['relativePath'] != key[1:]:
                    continue
                # 判断当前变异体是不是当前这个等级内的变异体，如果是就加进去
                for t in result[key][level]:
                    flag = False
                    if int(t) == item["linenum"]:
                        muInde.append(item["index"])
                        nowNum += 1
                        break
                if nowNum >= muNum:
                    return muInde
        level += 1
        if nowNum >= muNum or flag:
            break

    return muInde

muId = getMutantBasedSbfl()
# print(muId)
print(len(muId))

print('xsq')