import json
import logging
import multiprocessing
import os
import pickle
import sys
import paramiko
import random
import subprocess
from execute.FOM import executeFom, generateFom
from execute.SOM import executeSom, generateSom, subSom
from tool.cal_tools import calFomMbfl, countFunctionSus, calSomMbfl, countSOMFunctionSus
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
)
from tool.logger_config import logger_config
from tool.other import checkAndCreateDir, clearDir, run
from tool.remote_transmission import cp_from_remote, get_host_ip, ip, sftp_upload
from datetime import datetime


def getSbflSus(project, version):
    """
    获取sbfl的怀疑度值
    :param project: 项目名
    :param version: 版本号
    :return: 错误行信息和怀疑度列表
    suspiciousSbfl存储格式:
    错误文件路径: {
        sbfl方法名:{
            行号: 怀疑度
            ...
            行号: 怀疑度
        }
    }
    faultLocalization存储格式:
    错误文件路径: [行号, ..., 行号]
    }
    """
    try:
        suspiciousSbflPath = os.path.join(
            faultlocalizationResultPath, project, version, "suspiciousSbfl.json"
        )
        faultLocalizationPath = os.path.join(
            faultlocalizationResultPath, project, version, "faultLocalization.json"
        )
        if not os.path.exists(suspiciousSbflPath) or not os.path.exists(
            faultLocalizationPath
        ):
            print("\033[1;32m************** getSbflSus **************\033[0m")
            hugeToFilePath = os.path.join(
                outputCleanPath, project, version, "HugeToFile.txt"
            )
            with open(hugeToFilePath, "r") as f:
                hugeToFileList = f.readlines()
            hugeToFileList = [s.split("\t")[0] for s in hugeToFileList]
            delIndexPath = os.path.join(
                tpydataPath, project, version, "data_saveAgain_del_statement_index"
            )
            with open(delIndexPath, "rb") as f:
                delIndexList = pickle.load(f)
            faultPlusHugePath = os.path.join(
                outputCleanPath, project, version, "faultPlusHuge.in"
            )
            with open(faultPlusHugePath, "rb") as f:
                faultLineDic = pickle.load(f)
            susScorePath = os.path.join(tpydataPath, project, version, "sus_score")
            with open(susScorePath, "rb") as f:
                susScoreList = pickle.load(f)
            faultFilesLine = dict()
            for fault in faultLineDic.keys():
                fileLineNum = list()
                for index in range(0, len(hugeToFileList)):
                    if hugeToFileList[index] in fault:
                        fileLineNum.append(index)
                faultFilesLine[fault] = fileLineNum
            faultSbflSus = dict()
            for num in faultFilesLine.keys():
                # print(num)
                sbflSus = dict()
                for item in sbflMethod:
                    sbflSus[item] = dict()
                    faultSbflSus[num] = dict()
                t = 0
                distance = 0
                tFlag = True
                tFlag2 = True
                for index in range(0, len(hugeToFileList)):
                    if index in faultFilesLine[num] and tFlag:
                        distance = index
                        tFlag = False
                        for i in range(0, len(faultLineDic[num])):
                            faultLineDic[num][i] = faultLineDic[num][i] - distance

                    if delIndexList[index] is False:
                        if index in faultFilesLine[num]:
                            for item in sbflMethod:
                                sbflSus[item][index - distance] = susScoreList[item][t]
                            tFlag2 = False
                        elif tFlag2 is False:
                            break
                        t += 1
                for method in list(sbflSus.keys()):
                    for key in list(sbflSus[method].keys()):
                        if sbflSus[method][key] == 0:
                            del sbflSus[method][key]
                    faultSbflSus[num][method] = dict(
                        sorted(
                            sbflSus[method].items(), key=lambda x: x[1], reverse=True
                        )
                    )
            checkAndCreateDir(os.path.join(faultlocalizationResultPath))
            checkAndCreateDir(os.path.join(faultlocalizationResultPath, project))
            checkAndCreateDir(
                os.path.join(faultlocalizationResultPath, project, version)
            )
            with open(suspiciousSbflPath, "w") as f:
                f.write(json.dumps(faultSbflSus, indent=2))
            with open(faultLocalizationPath, "w") as f:
                f.write(json.dumps(faultLineDic, indent=2))
        with open(suspiciousSbflPath, "r") as f:
            faultSbflSus = json.load(f)
        with open(faultLocalizationPath, "r") as f:
            faultLineDic = json.load(f)
        if ip != "202.4.130.30":
            sftp_upload(
                "202.4.130.30",
                "fanluxi",
                password,
                suspiciousSbflPath,
                suspiciousSbflPath,
            )
        if ip != "202.4.130.30":
            sftp_upload(
                "202.4.130.30",
                "fanluxi",
                password,
                faultLocalizationPath,
                faultLocalizationPath,
            )

        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath))
        checkAndCreateDir(os.path.join(SOMfaultlocalizationResultPath, project))
        checkAndCreateDir(
            os.path.join(SOMfaultlocalizationResultPath, project, version)
        )
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")
        return
    print("\033[1;32m************** getSbflSus SUCCESS **************\033[0m")
    return faultLineDic, faultSbflSus


# 一阶变异体
def FOM(project, version, configData):
    # print(project, version)
    logging.info(project + " " + version)
    try:
        faultLineDic, sbflSus = getSbflSus(project, version)
        # print(faultLineDic)
        logging.info(faultLineDic)
        muInfoList = generateFom(project, version)
        resultList = executeFom(project, version, muInfoList, configData)
        calFomMbfl(project, version, muInfoList, resultList)
        # countFunctionSus(project, version)
        logging.info(project + " " + version + " success!")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")
        return


# 二阶变异体
def SOM(project, version, configData):
    # print(project, version)
    logging.info(project + " " + version)
    try:
        faultLineDic, sbflSus = getSbflSus(project, version)
        # print(faultLineDic)
        # logging.info(faultLineDic)
        muInfoList, somInfoList = generateSom(project, version)
        
        # 异步执行pkill命令
        pkill_process = subprocess.Popen(['pkill', '-9', '-u', 'fanluxi', 'java'], stderr=subprocess.PIPE)

        # 等待pkill命令完成
        pkill_process.wait()
        
        subSOMInfoList = subSom(
            project, version, muInfoList, somInfoList, configData, faultLineDic
        )
        
        
        # 异步执行pkill命令
        pkill_process = subprocess.Popen(['pkill', '-9', '-u', 'fanluxi', 'java'], stderr=subprocess.PIPE)

        # 等待pkill命令完成
        pkill_process.wait()
        
        resultList = executeSom(project, version, subSOMInfoList, configData)
        
        # 异步执行pkill命令
        pkill_process = subprocess.Popen(['pkill', '-9', '-u', 'fanluxi', 'java'], stderr=subprocess.PIPE)

        # 等待pkill命令完成
        pkill_process.wait()
        
        calSomMbfl(project, version, subSOMInfoList, resultList, "SOM.json")
        logging.info(project + " " + version + " success!")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")
        return

# new二阶变异体（对newsubSOMInfo.json进行计算）
def newSOM(project, version, configData):
    muInfoPath = f"{faultlocalizationResultPath}/{project}/{version}/muInfo.json"
    with open(muInfoPath, "r") as f:
        muInfo = json.load(f)

    muResultPath = f"{faultlocalizationResultPath}/{project}/{version}/muResult.json"   
    with open(muResultPath, "r") as f:
        muResult = json.load(f)
    # print(project, version)
    # logging.info(project + " " + version)
    try:
        with open(os.path.join(SOMfaultlocalizationResultPath, project, version, "newsubSOMInfo.json"), 'r') as f:
            newsubSOMInfoList = json.load(f)
        
        newresultList = executeSom(project, version, newsubSOMInfoList, configData)
        
        # 异步执行pkill命令
        pkill_process = subprocess.Popen(['pkill', '-9', '-u', 'fanluxi', 'java'], stderr=subprocess.PIPE)

        # 等待pkill命令完成
        pkill_process.wait()
        
        calSomMbfl(project, version, newsubSOMInfoList, newresultList, "newSOM.json", muInfo, muResult)
        # logging.info(project + " " + version + " success!")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")
        return



def SGS(project, version, configData, count_SGS):
    # logging.info(project + " " + version)
    try:
        faultLineDic, sbflSus = getSbflSus(project, version)
        muInfoList, somInfoList = generateSom(project, version)
        subSOMInfoList = subSom(
            project, version, muInfoList, somInfoList, configData, faultLineDic
        )
        resultList = executeSom(project, version, subSOMInfoList, configData)

        groups = {}
        for item in resultList:
            linenum1 = item["linenum1"]
            linenum2 = item["linenum2"]
            
            if linenum1 not in groups:
                groups[linenum1] = []
            groups[linenum1].append(item)
            
            if linenum2 not in groups:
                groups[linenum2] = []
            groups[linenum2].append(item)
        
        for i in range(1,6):
            # 新的集合
            new_groups = {}

            # 遍历每个组
            for key, values in groups.items():
                num_elements = max(1, round(len(values) * i / 10))
                selected_elements = random.sample(values, num_elements)
                new_groups[key] = selected_elements

            newresultList = []
            seen_indexes = set()
            for key, values in new_groups.items():
                for value in values:
                    # newresultList.append(value)
                    if value['index'] not in seen_indexes:
                        newresultList.append(value)
                        seen_indexes.add(value['index'])
            
            newsubSOMList = []
            for item in newresultList:
                for value in subSOMInfoList:
                    if value['index'] == item['index']:
                        newsubSOMList.append(value)
                        break
            
            count_SGS[i] += len(newsubSOMList)
            logging.info(f"SGS{i}:{count_SGS[i]}")
            # logging.info(len(seen_indexes))
            # calSomMbfl(project, version, newsubSOMList, newresultList, f"SGS-{i}0%.json")
        return count_SGS
            

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")
        return
    
def SHMR(project, version, configData):
    """
    基于SBFL对SGS进行改进，只要在sbfl中前百分之70的语句生成的变异体，如果某一行怀疑度在SBFL中属于后30%，则直接抛弃该行生成的变异体
    """
    # logging.info(project + " " + version)
    try:
        faultLineDic, sbflSus = getSbflSus(project, version)
        muInfoList, somInfoList = generateSom(project, version)
        subSOMInfoList = subSom(
            project, version, muInfoList, somInfoList, configData, faultLineDic
        )
        resultList = executeSom(project, version, subSOMInfoList, configData)
        muInfoPath = f"{faultlocalizationResultPath}/{project}/{version}/muInfo.json"
        with open(muInfoPath, "r") as f:
            muInfo = json.load(f)

        muResultPath = f"{faultlocalizationResultPath}/{project}/{version}/muResult.json"
        
        with open(muResultPath, "r") as f:
            muResult = json.load(f)

        # 初始化存储结果的字典
        top_50_percent_lines = {}
        for file_path, metrics in sbflSus.items():
            ochiai_values = metrics.get("ochiai", {})
            sorted_lines = sorted(ochiai_values, key=ochiai_values.get, reverse=True)
            half_length = (int)(len(sorted_lines) * 0.8)
            top_lines = sorted_lines[:half_length]
            top_lines = [int(line) for line in top_lines]
            top_50_percent_lines[file_path] = top_lines
        # print(top_50_percent_lines)
        groups = {}
        for item in subSOMInfoList:
            linenum1 = item["linenum1"]
            linenum2 = item["linenum2"]
            flag = True
            for key, values in top_50_percent_lines.items():
                if item['relativePath1'] in key and item['linenum1'] in values:
                    flag = False
                    break
                if item['relativePath2'] in key and item['linenum2'] in values:
                    flag = False
                    break
            if flag:
                continue
            # print(item)
            if linenum1 not in groups:
                groups[linenum1] = []
            groups[linenum1].append(item)
            
            if linenum2 not in groups:
                groups[linenum2] = []
            groups[linenum2].append(item)
        
        for i in range(3,5):
            new_muInfo = random.sample(muInfo, round(len(resultList) * i * 0.01))
            new_muResult = []
            for item in new_muInfo:
                for value in muResult:
                    if value['index'] == item['index']:
                        new_muResult.append(value)
                        break
        
            # 新的集合
            new_groups = {}

            # 遍历每个组
            for key, values in groups.items():
                num_elements = max(1, round(len(values) * i * 0.01))
                selected_elements = random.sample(values, num_elements)
                new_groups[key] = selected_elements

            newsubSOMList = []
            for key, values in new_groups.items():
                for value in values:
                    newsubSOMList.append(value)
            
            newsubSOMList = random.sample(newsubSOMList,min(round(len(resultList) * 0.01 * i), len(newsubSOMList)) )
            
            newresultList = []
            for item in newsubSOMList:
                for value in resultList:
                    if value['index'] == item['index']:
                        newresultList.append(value)
                        break
            # logging.info(len(newsubSOMList))
            calSomMbfl(project, version, newsubSOMList, newresultList, f"SHMSR-{i}%.json", new_muInfo, new_muResult)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")
        return

def SAMPLING(project, version, configData):
    # logging.info(project + " " + version)
    try:
        faultLineDic, sbflSus = getSbflSus(project, version)
        muInfoList, somInfoList = generateSom(project, version)
        subSOMInfoList = subSom(
            project, version, muInfoList, somInfoList, configData, faultLineDic
        )
        resultList = executeSom(project, version, subSOMInfoList, configData)
        muInfoPath = f"{faultlocalizationResultPath}/{project}/{version}/muInfo.json"
        with open(muInfoPath, "r") as f:
            muInfo = json.load(f)

        muResultPath = f"{faultlocalizationResultPath}/{project}/{version}/muResult.json"
        
        with open(muResultPath, "r") as f:
            muResult = json.load(f)

        sample_rates = [3, 4, 5, 10, 15] # 采样率设置为3%，4%，5%，10%，15%
        
        for i in sample_rates:
            new_muInfo = random.sample(muInfo, round(len(resultList) * i * 0.01))
            new_muResult = []
            for item in new_muInfo:
                for value in muResult:
                    if value['index'] == item['index']:
                        new_muResult.append(value)
                        break
        
            newresultList = random.sample(resultList, round(len(resultList) * i * 0.01))
        
            newsubSOMList = []
            for item in newresultList:
                for value in subSOMInfoList:
                    if value['index'] == item['index']:
                        newsubSOMList.append(value)
                        break
            calSomMbfl(project, version, newsubSOMList, newresultList, f"SOM-SAMPLING-{i}%.json", new_muInfo, new_muResult)
            

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")

def SELECTIVE(project, version, configData):
    # logging.info(project + " " + version)
    try:
        faultLineDic, sbflSus = getSbflSus(project, version)
        muInfoList, somInfoList = generateSom(project, version)
        subSOMInfoList = subSom(
            project, version, muInfoList, somInfoList, configData, faultLineDic
        )
        resultList = executeSom(project, version, subSOMInfoList, configData)
        
        mutationOperator = ['ORU', 'LOR', 'COR', 'LVR', 'STD', 'SOR', 'ROR', 'AOR']
        muInfoPath = f"{faultlocalizationResultPath}/{project}/{version}/muInfo.json"
        with open(muInfoPath, "r") as f:
            muInfo = json.load(f)

        muResultPath = f"{faultlocalizationResultPath}/{project}/{version}/muResult.json"
        
        with open(muResultPath, "r") as f:
            muResult = json.load(f)
            
        sample_rates = [3, 4, 5, 10, 15] # 采样率设置为3%，4%，5%，10%，15%
        for i in sample_rates:
            new_muInfo = []
            new_muResult = []
            submutationOperator = random.sample(mutationOperator, 1)

            newsubSOMList = []
            for item in subSOMInfoList:
                if item['typeOp1'] in submutationOperator and item['typeOp2'] in submutationOperator:
                    newsubSOMList.append(item)
            newsubSOMList = random.sample(newsubSOMList,min(round(len(resultList) * 0.01 * i), len(newsubSOMList)) )
            
            newresultList = []
            for item in newsubSOMList:
                for value in resultList:
                    if value['index'] == item['index']:
                        newresultList.append(value)
                        break
            
            
            calSomMbfl(project, version, newsubSOMList, newresultList, f"SOM-SELECTIVE-{i}%_2.json", new_muInfo, new_muResult)
            
            

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")


if __name__ == "__main__":
    clearDir(tempSrcPath)
    checkAndCreateDir(tempSrcPath)
    logger = logger_config(log_path="logs/main.log")
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    # 10b XSQ
    # with open("./failVersion.json", "r") as f:
    #     failVersion = json.load(f)
    # with open("./mutilFaultVersion.json", "r") as f:
    #     mutilFaultVersion = json.load(f)
    with open("./config.json", "r") as configFile:
        configData = json.load(configFile)
    try:
        start_time = datetime.now()
        with multiprocessing.Pool(12) as executor:
            for projectDir in project.keys():
                count = 0
                projectDir = "Closure"
                # projectDir = "Time"
                # projectDir = "Cli"
                # projectDir = "Chart"
                # projectDir = "Closure"
                # projectDir = "Math"
                # projectDir = "Mockito"
                # projectDir = "JacksonDatabind"
                # projectDir = "JxPath"
                for versionNum in range(1, project[projectDir] + 1):
                    versionDir = str(versionNum) + "b"
                    versionDir = "115b"
                    # if versionDir == '12b':
                    #     exit(1)
                    # if (
                    #     not failVersion.get(projectDir) is None
                    #     and versionDir in failVersion[projectDir]
                    # ):
                    #     continue
                    # if ip != "202.4.130.30":
                    #     clearDir(djSrcPath)
                    #     clearDir(outputCleanPath)
                    #     clearDir(tpydataPath)
                    #     clearDir(faliingTestOutputPath)
                    #     ssh = paramiko.SSHClient()
                    #     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    #     ssh.connect("202.4.130.30", username="fanluxi", password=password)
                    #     sftp = ssh.open_sftp()
                    #     try:
                    #         sftp.stat(
                    #             os.path.join(
                    #                 faultlocalizationResultPath, projectDir, versionDir
                    #             )
                    #         )
                    #     except FileNotFoundError:
                    #         try:
                    #             sftp.stat(
                    #                 os.path.join(faultlocalizationResultPath, projectDir)
                    #             )
                    #         except FileNotFoundError:
                    #             sftp.mkdir(
                    #                 os.path.join(faultlocalizationResultPath, projectDir)
                    #             )
                    #         finally:
                    #             sftp.mkdir(
                    #                 os.path.join(
                    #                     faultlocalizationResultPath, projectDir, versionDir
                    #                 )
                    #             )
                    #         if cp_from_remote(projectDir, versionDir):
                    #             FOM(projectDir, versionDir, configData)
                    #             # SOM(projectDir, versionDir)
                    #     finally:
                    #         sftp.close()
                    #         ssh.close()
                    # elif not os.path.exists(os.path.join(faultlocalizationResultPath, projectDir, versionDir)):
                    # try:
                    #     os.makedirs(os.path.join(
                    #         faultlocalizationResultPath, projectDir, versionDir))
                    # finally:
                    # if mutilFaultVersion.get(projectDir) is None or versionDir not in mutilFaultVersion[projectDir]:
                    #     continue
                    # current_time = datetime.now()
                    # formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    # print(formatted_time)
                    # FOM(projectDir, versionDir, configData)
                    # SGS(projectDir, versionDir, configData)
                    newSOM(projectDir, versionDir, configData)
                    # executor.apply_async(newSOM, (projectDir, versionDir,configData))
                    
                    # SHMR(projectDir, versionDir, configData)
                    # SAMPLING(projectDir, versionDir, configData)
                    # SELECTIVE(projectDir, versionDir, configData)
                    # exit(1)
            # executor.close()
            # executor.join()
            
        end_time = datetime.now()
        mutate_duration = (end_time - start_time).total_seconds()  # 计算变异时间
        hours = int(mutate_duration // 3600)  # 一小时3600秒
        minutes = int((mutate_duration % 3600) // 60)
        seconds = mutate_duration % 60
        print(f"变异操作耗时 {hours} 小时 {minutes} 分钟 {seconds:.2f} 秒")
        logging.info(projectDir + " " + versionDir + " " + f"变异操作耗时 {hours} 小时 {minutes} 分钟 {seconds:.2f} 秒")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        print(f"\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m")
        logging.error(f"Error in {file_name} at line {line_number}: {e}")
        

