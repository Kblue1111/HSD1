import os
import shutil
import operator
import json
import math
import execute.SOMExecutorTool as SOMExecutorTool
import concurrent.futures
import sys
import random
import logging
from tool.config_variables import (
    tempSrcPath,
    tpydataPath,
    outputCleanPath,
    djSrcPath,
    mutantsFilePath,
    faliingTestOutputPath,
    faultlocalizationResultPath,
    SOMfaultlocalizationResultPath,
    sbflMethod,
    sourcePath,
    password,
    project,
)
from tool.remote_transmission import ip, get_host_ip, sftp_upload, cp_from_remote
from tool.logger_config import logger_config
from tool.mbfl_formulas import (
    dstar,
    ochiai,
    gp13,
    op2,
    jaccard,
    russell,
    turantula,
    naish1,
    binary,
    crosstab,
)
from tool.other import clearDir, checkAndCreateDir, run
from tool.cal_tools import calFomMbfl, countFunctionSus

from tool.config_variables import (
    SOMfaultlocalizationResultPath,
    djSrcPath,
    faliingTestOutputPath,
    faultlocalizationResultPath,
    mbflMethods,
    mutantsFilePath,
    outputCleanPath,
    password,
    project,
    sbflMethod,
    sourcePath,
    tempSrcPath,
    tpydataPath,
    method_names,
)


def generateSom(project, version) -> list:
    """
    通过major获取变异体信息
    somInfo存储格式:
    index: 变异体序号
    linenum1: 变异体1行号
    linenum2: 变异体2行号
    typeOp1: 变异算子类型1
    typeOp2: 变异算子类型2
    mutFilePath1: 变异体1存储位置
    mutFilePath2: 变异体2存储位置
    relativePath1: 变异体1文件在项目中的相对路径
    relativePath2: 变异体2文件在项目中的相对路径
    """
    try:
        clearDir("./tmp")
        # 变异体文件存储位置
        mutantPath = os.path.join(mutantsFilePath, project, version)
        # if not os.path.exists(mutantPath):
        #     print("\033[1;32m************** generateFom **************\033[0m")
        #     shutil.copytree(os.path.join(djSrcPath, project, version), "./tmp")
        #     run("./runMajor.sh")
        #     shutil.copytree("./tmp/mutants", mutantPath)
        #     shutil.copyfile("./tmp/mutants.log", mutantPath + "/mutants.log")
        # 变异体信息存储位置
        muInfoPath = os.path.join(
            SOMfaultlocalizationResultPath, project, version, "muInfo.json"
        )
        if not os.path.exists(muInfoPath):
            muInfoList = list()
            with open(mutantPath + "/mutants.log", "r") as f:
                for line in f.readlines():
                    muInfo = dict()
                    muInfo["index"] = int(line.split(":")[0])
                    muInfo["linenum"] = int(line.split(":")[5])
                    muInfo["typeOp"] = line.split(":")[1]
                    muInfoList.append(muInfo)
            for i in os.listdir(mutantPath):
                # 找到以序号为名的文件夹， 除去mutants.log
                if os.path.isdir(os.path.join(mutantPath, i)):
                    mutFileDir = os.listdir(os.path.join(mutantPath, i))[0]
                    mutFilePath = os.path.join(mutantPath, i, mutFileDir)
                    if len(sourcePath[project].keys()) > 1 and int(version[:-1]) > int(
                        list(sourcePath[project].keys())[0]
                    ):
                        relativePath = os.path.join(
                            sourcePath[project][list(sourcePath[project].keys())[1]],
                            mutFileDir,
                        )
                    else:
                        relativePath = os.path.join(
                            sourcePath[project][list(sourcePath[project].keys())[0]],
                            mutFileDir,
                        )
                    # 递归找到文件
                    while os.path.isdir(mutFilePath):
                        mutFileDir = os.listdir(mutFilePath)[0]
                        mutFilePath = os.path.join(mutFilePath, mutFileDir)
                        relativePath += "/" + mutFileDir
                    muInfoList[int(i) - 1]["mutFilePath"] = mutFilePath
                    muInfoList[int(i) - 1]["relativePath"] = relativePath
            with open(muInfoPath, "w") as f:
                f.write(json.dumps(muInfoList, indent=2))
        with open(muInfoPath, "r") as f:
            muInfoList = json.load(f)
        if ip != "202.4.130.30":
            sftp_upload("202.4.130.30", "fanluxi", password, muInfoPath, muInfoPath)

        # SOM信息存储位置
        somInfoPath = os.path.join(
            SOMfaultlocalizationResultPath, project, version, "SOMInfo.json"
        )
        if not os.path.exists(somInfoPath):
            somInfoList = list()
            num = 1
            for i, item1 in enumerate(muInfoList):
                for j, item2 in enumerate(muInfoList):
                    if i >= j:
                        continue
                    if item1["relativePath"] != item2["relativePath"]:
                        continue
                    somInfo = dict()
                    somInfo["index"] = num
                    num += 1
                    somInfo["linenum1"] = item1["linenum"]
                    somInfo["linenum2"] = item2["linenum"]
                    somInfo["typeOp1"] = item1["typeOp"]
                    somInfo["typeOp2"] = item2["typeOp"]
                    somInfo["mutFilePath1"] = item1["mutFilePath"]
                    somInfo["mutFilePath2"] = item2["mutFilePath"]
                    somInfo["relativePath1"] = item1["relativePath"]
                    somInfo["relativePath2"] = item2["relativePath"]
                    somInfoList.append(somInfo)
            with open(somInfoPath, "w") as f:
                f.write(json.dumps(somInfoList, indent=2))
        with open(somInfoPath, "r") as f:
            somInfoList = json.load(f)
        if ip != "202.4.130.30":
            sftp_upload("202.4.130.30", "fanluxi", password, somInfoPath, somInfoPath)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f"\033[1;31mError at line {line_number}: {e}\033[0m")
        return None
    print("\033[1;32m************** generateSom SUCCESS **************\033[0m")
    return muInfoList, somInfoList


def calSus(resultList):
    susResult = []
    if resultList == None:
        resultList = []
    for i in range(0, len(resultList)):
        if resultList[i]["status"] == 0:
            continue
        Anp = 0
        Anf = 0
        Akp = 0
        Akf = 0
        for index in range(0, len(resultList[i]["passList"]["type1"])):
            if resultList[i]["passList"]["type2"][index] == 1:
                if resultList[i]["killList"]["type2"][index] == 1:
                    Akf += 1
                else:
                    Anf += 1
            else:
                if resultList[i]["killList"]["type2"][index] == 1:
                    Akp += 1
                else:
                    Anp += 1
        susResult.append(ochiai(Akf, Anf, Akp, Anp))
    return susResult


def subSom(project, version, muInfoList, somInfoList, configData, faultLineDic):
    try:
        # 变异体信息存储位置
        subSOMInfoPath = os.path.join(
            SOMfaultlocalizationResultPath, project, version, "subSOMInfo.json"
        )
        if not os.path.exists(subSOMInfoPath):
            faultSOMList1 = list()
            faultSOMList2 = list()
            sum_value = 0
            for key, value in faultLineDic.items():
                sum_value += len(value)
            sample_size = int(len(muInfoList)/sum_value)
            for key, value in faultLineDic.items():
                for v in value:
                    SOMList = list()
                    for i, item in enumerate(somInfoList):
                        if (item["relativePath1"] in key and item["linenum1"] == v) or (
                            item["relativePath2"] in key and item["linenum2"] == v
                        ):
                            SOMList.append(item)
                    resultList = list()
                    if SOMList == None:
                        SOMList = []
                    if not isinstance(sample_size, int) or sample_size <= 0 or sample_size > len(SOMList):
                        SOMList = SOMList
                    else:
                        SOMList = random.sample(SOMList, sample_size)
                    for item in SOMList:
                        item["project"] = project
                        item["version"] = version
                    # 创建线程池
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=16
                    ) as executor:
                        future_results = [
                            executor.submit(execute_som, muInfo, configData)
                            for muInfo in SOMList
                        ]
                        for future_result in future_results:
                            try:
                                result = future_result.result(5 * 60)
                                resultList.append(result)
                            except concurrent.futures.TimeoutError:
                                print("Task timed out and will be terminated")
                                # 尝试取消任务
                                if not future_result.cancel():
                                    print("任务无法被取消，可能已经完成或正在进行中")
                                else:
                                    print("任务成功被取消")
                                # executor.shutdown(wait=False)
                                # break
                            except KeyboardInterrupt:
                                print("Interrupted by user and will be terminated")
                                executor.shutdown(wait=False)
                                exit(1)
                    susResult = calSus(resultList)
                    if susResult:
                        # 获取最大值的下标
                        max_index = susResult.index(max(susResult))
                        max_SOM_value = SOMList[max_index]
                        faultSOMList1.append(max_SOM_value)
                        # 将 SOMList 转换成集合，并去除 max_SOM_value
                        SOMSet = [item for item in SOMList if item != max_SOM_value]

                        # 将剩余的元素加入到 faultSOMList2
                        faultSOMList2.extend(SOMSet)
                        # print(faultSOMList1)
                        # print(faultSOMList2)

            subSOMInfoList = list()
            subSOMInfoList.extend(faultSOMList1)

            elements_to_add = len(muInfoList) - len(subSOMInfoList)
            subSOMInfoList.extend(random.sample(somInfoList, elements_to_add))
            with open(subSOMInfoPath, "w") as f:
                f.write(json.dumps(subSOMInfoList, indent=2))
        with open(subSOMInfoPath, "r") as f:
            subSOMInfoList = json.load(f)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f"\033[1;31mError at line {line_number}: {e}\033[0m")
        return None

    return subSOMInfoList


def execute_som(somInfo, configData):
    print("变异体编号:", somInfo["index"])
    SOMexecuteInfoPath = os.path.join(
        SOMfaultlocalizationResultPath,
        somInfo["project"],
        somInfo["version"],
        "executeInfo",
        str(somInfo["index"]),
    )
    if not os.path.exists(SOMexecuteInfoPath):
        executor = SOMExecutorTool.secondOrderExecutorTool(
            somInfo["project"], somInfo["version"], somInfo, configData
        )
        somExecutResult = {
            "index": somInfo["index"],
            "linenum1": somInfo["linenum1"],
            "linenum2": somInfo["linenum2"],
            "status": executor.status,
            "passList": executor.passList,
            "killList": executor.killList,
            "mKillList": executor.mKillList,
        }
        checkAndCreateDir(
            os.path.join(
                SOMfaultlocalizationResultPath,
                somInfo["project"],
                somInfo["version"],
                "executeInfo",
            )
        )
        with open(SOMexecuteInfoPath, "w") as f:
            f.write(json.dumps(somExecutResult, indent=2))
    else:
        with open(SOMexecuteInfoPath, "r") as f:
            somExecutResult = json.load(f)
    return somExecutResult


def executeSom(project, version, muInfoList, configData):
    """
    执行二阶变异体
    muResult存储格式:
    index: 变异体序号
    linenum: 变异体行号
    status: 执行结果(0为执行失败,1为执行成功)
    passList: 执行结果
    killList: 杀死信息
    mKillList: MUSE版杀死信息
    """
    try:
        somResultPath = os.path.join(
            SOMfaultlocalizationResultPath, project, version, "newsomResult.json"
        )
        if not os.path.exists(somResultPath):
            print("\033[1;32m************** executeSOM **************\033[0m")
            resultList = list()
            if muInfoList == None:
                muInfoList = []
            for item in muInfoList:
                item["project"] = project
                item["version"] = version
            # 创建线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                future_results = [
                    executor.submit(execute_som, muInfo, configData)
                    for muInfo in muInfoList
                ]

                for future_result in future_results:
                    try:
                        result = future_result.result(5 * 60)
                        resultList.append(result)
                    except concurrent.futures.TimeoutError:
                        print("Task timed out and will be terminated")
                        # executor.shutdown(wait=False)
                        # break
                    except KeyboardInterrupt:
                        print("Interrupted by user and will be terminated")
                        executor.shutdown(wait=False)
                        exit(1)
            with open(somResultPath, "w") as f:
                f.write(json.dumps(resultList, indent=2))
        with open(somResultPath, "r") as f:
            resultList = json.load(f)
        if ip != "202.4.130.30":
            sftp_upload(
                "202.4.130.30", "fanluxi", password, somResultPath, somResultPath
            )
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f"\033[1;31mError at line {line_number}: {e}\033[0m")
        logging.error(f"Error at line {line_number}: {e}")
        return
    # print("\033[1;32m************** executesubSom SUCCESS **************\033[0m")
    return resultList
