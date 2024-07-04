from tool.cal_tools import calFomMbfl, countFunctionSus
import os
import shutil
import json
import concurrent.futures
import sys
import logging
import execute.FOMExecutorTool as FOMExecutorTool
from tool.config_variables import tempSrcPath, tpydataPath, outputCleanPath, djSrcPath, mutantsFilePath, faliingTestOutputPath, faultlocalizationResultPath, SOMfaultlocalizationResultPath, sbflMethod, sourcePath, password, project
from tool.remote_transmission import ip, get_host_ip, sftp_upload, cp_from_remote
from tool.logger_config import logger_config
from tool.mbfl_formulas import dstar, ochiai, gp13, op2, jaccard, russell, turantula, naish1, binary, crosstab
from tool.other import clearDir, checkAndCreateDir, run

muInfoPath = os.path.join(
    faultlocalizationResultPath, "Chart", "1b", "muInfo.json")
with open(muInfoPath, 'r') as f:
    muInfoList = json.load(f)
muResultPath = os.path.join(
    faultlocalizationResultPath, "Chart", "1b", "muResult.json")
with open(muResultPath, 'r') as f:
    resultList = json.load(f)
calFomMbfl("Chart", "1b", muInfoList, resultList)