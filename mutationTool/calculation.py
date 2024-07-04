import json
import logging
import os
import pickle
import sys
import concurrent
import time
import multiprocessing
import paramiko
from execute.FOM import executeFom, generateFom
from execute.SOM import executeSom, generateSom
from tool.cal_tools import calSbfl, countTotal, calSOM, calEXAM, countTotalEXAM, calSOM_czx, calSOM_czx2, calSOM_czx3, calSOM_czx4, calSOM_czx5, calFomMbfl, calTopNMbflAverage, calTopNMbflBest, calTopNMbflWorst, countFunctionSus, countSOMFunctionSus, countTotalMARandMFR
from tool.config_variables import (SOMfaultlocalizationResultPath, djSrcPath,
                                   faliingTestOutputPath,
                                   faultlocalizationResultPath, mbflMethods,
                                   mutantsFilePath, outputCleanPath, password,
                                   project, sbflMethod, sourcePath,
                                   tempSrcPath, tpydataPath)
from tool.count_data import countTonN, countMARandMFR, countTonN_Mutil, countEXAM
from tool.logger_config import logger_config
from tool.mbfl_formulas import (binary, crosstab, dstar, gp13, jaccard, naish1,
                                ochiai, op2, russell, turantula)
from tool.other import checkAndCreateDir, clearDir, run
from tool.remote_transmission import (cp_from_remote, get_host_ip, ip,
                                      sftp_upload)
from main import SGS, SHMR, SAMPLING, SELECTIVE

if __name__ == '__main__':
    with open("./failVersion.json", "r") as f:
        failVersion = json.load(f)
    with open("./mutilFaultVersion.json", "r") as f:
        mutilFaultVersion = json.load(f)
    with open("./config.json", "r") as configFile:
        configData = json.load(configFile)
    logger = logger_config(log_path="logs/calculation.log")
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    
    # 统计高阶变异体数量
    # count = 0
    # for projectDir in project.keys():
    #     for versionNum in range(1, project[projectDir] + 1):
    #         versionDir = str(versionNum) + 'b'
    #         # if not failVersion.get(projectDir) is None and versionDir in failVersion[projectDir]:
    #         #     continue
    #         if mutilFaultVersion.get(projectDir) is None or versionDir not in mutilFaultVersion[projectDir]:
    #             continue
    #         with open(os.path.join(SOMfaultlocalizationResultPath, projectDir, versionDir, "muInfo.json"), "r") as f:
    #             data = json.load(f)
    #         count += len(data)
    # print(count)
    
    with multiprocessing.Pool(60) as executor:
        for projectDir in project.keys():
            # projectDir = "Chart"
            for versionNum in range(1, project[projectDir] + 1):
                versionDir = str(versionNum) + 'b'
        #         # versionDir = "11b"
        #         # if not failVersion.get(projectDir) is None and versionDir in failVersion[projectDir]:
        #         #     continue
                if mutilFaultVersion.get(projectDir) is None or versionDir not in mutilFaultVersion[projectDir]:
                    continue
                print(projectDir, versionDir)
                    
        #         # SBFL方法级错误定位
        #         # executor.apply_async(calSbfl, (projectDir, versionDir))
                
                
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_randomForestModel_10_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmt_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmt_3.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmt_4.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmt_5.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_naiveBayesModel_10_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmbfl_3_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmbfl_4_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmbfl_5_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmbfl_10_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmbfl_15_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_logisticRegressionModel_15_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_randomForestModel_15_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmbfl_4_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "predSuspicious", "pred_susFunc_pmbfl_5_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SHMSR-3%.json")) 
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SHMSR-4%.json")) 
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SHMSR-10%.json")) 
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SHMSR-20%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SHMSR-30%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SHMSR-40%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SHMSR-50%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SAMPLING-3%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SAMPLING-4%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SAMPLING-10%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SAMPLING-20%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SAMPLING-30%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SAMPLING-40%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SAMPLING-50%.json"))
                executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SELECTIVE-3%_2.json"))
                executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SELECTIVE-4%_2.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SELECTIVE-10%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SELECTIVE-20%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SELECTIVE-30%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SELECTIVE-40%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM-SELECTIVE-50%.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "SOM_czx3.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "sbfl.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "hmer.json"))
                # executor.apply_async(calEXAM, (projectDir, versionDir, "susFunction", "complete.json"))
                # 计算高阶变异体语句级怀疑度
                # executor.apply_async(calSOM, (projectDir, versionDir))
                # executor.apply_async(calSOM_czx3, (projectDir, versionDir))
                # executor.apply_async(calSOM_czx5, (projectDir, versionDir))
                
                # # 计算SGS
                # executor.apply_async(SGS, (projectDir, versionDir,configData))
                
                # # 计算SHMR
                # executor.apply_async(SHMR, (projectDir, versionDir,configData))
                
                # # 计算高阶版SAMPLING
                # executor.apply_async(SAMPLING, (projectDir, versionDir,configData))
                
                # # 计算高阶版SELECTIVE
                # executor.apply_async(SELECTIVE, (projectDir, versionDir,configData))
                
                
                # 遍历susStatement目录，计算方法级怀疑度
                # executor.apply_async(countSOMFunctionSus, (projectDir, versionDir))
            # break
                
        executor.close()
        executor.join()
        for projectDir in project.keys():
            # countTonN_Mutil(SOMfaultlocalizationResultPath, projectDir, "TopNFunctionWorst", mutilFaultVersion, "TopNFunctionWorstMutilFault")
        #     countMARandMFR(SOMfaultlocalizationResultPath, projectDir, "Function_MAR")
        #     countMARandMFR(SOMfaultlocalizationResultPath, projectDir, "Function_MFR")
            countEXAM(SOMfaultlocalizationResultPath, projectDir, "EXAMsusFunction")
            # countEXAM(SOMfaultlocalizationResultPath, projectDir, "EXAMpredSuspicious")
        #     # break
            
        # countTotal("/home/fanluxi/pmbfl/SOMprocessedData/TopNFunctionWorstMutilFault")
        
        # countTotalMARandMFR("/home/fanluxi/pmbfl/SOMprocessedData/Function_MAR")
        # countTotalMARandMFR("/home/fanluxi/pmbfl/SOMprocessedData/Function_MFR")
        countTotalEXAM("/home/fanluxi/pmbfl/SOMprocessedData/EXAMsusFunction")
        # countTotalEXAM("/home/fanluxi/pmbfl/SOMprocessedData/EXAMpredSuspicious")
        
        # countTotal("/home/fanluxi/pmbfl/trainModelSOM/predTopN/predTopNFunctionWorst")
        
        # countTotalMARandMFR("/home/fanluxi/pmbfl/trainModelSOM/predMFR")
        # countTotalMARandMFR("/home/fanluxi/pmbfl/trainModelSOM/predMAR")
        # countTotalMARandMFR("/home/fanluxi/pmbfl/trainModelSOM/ModelMetrics")