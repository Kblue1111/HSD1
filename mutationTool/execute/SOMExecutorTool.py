import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import os
import shutil
import uuid
import json
import time
from os import system
from threading import Thread
import sys
import subprocess
import signal
import difflib
import fileinput
from tool.config_variables import tempSrcPath, tpydataPath, outputCleanPath, djSrcPath, mutantsFilePath, faliingTestOutputPath, faultlocalizationResultPath, SOMfaultlocalizationResultPath, sbflMethod, sourcePath, password, project

result = 0

# def compile(programeDir, self):
#     # print('开始编译')
#     # 执行d4j自带的编译脚本
#     cmd = ['{}/framework/bin/defects4j'.format(self.configData['D4jHome']), 'compile', '-w', programeDir]
#     completed_process = None
#     try:
#         with open('logs/d4jCompile.log', 'a') as log_file:
#             completed_process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True, timeout=120)

#     except subprocess.TimeoutExpired:
#         self.status = 0
#         print('编译超时，杀死进程')
#         if completed_process is not None:
#             completed_process.terminate()
#             completed_process.kill()
#         return False

#     except subprocess.CalledProcessError:
#         self.status = 0
#         print('编译失败')
#         return False

#     return True


# def test(programDir, self):
    # # 执行d4j自带的测试脚本
    # cmd = ['{}/framework/bin/defects4j'.format(self.configData['D4jHome']), 'test', '-w', programeDir]
    # completed_process = None
    # try:
    #     with open('logs/d4jTest.log', 'a') as log_file:
    #         completed_process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True, timeout=120)

    # except subprocess.TimeoutExpired:
    #     self.status = 0
    #     print('测试超时，杀死进程')
    #     if completed_process is not None:
    #         completed_process.terminate()
    #         completed_process.kill()
    #     # os.killpg(os.getpgid(completed_process.pid), signal.SIGTERM)
    #     return False

    # except subprocess.CalledProcessError:
    #     self.status = 0
    #     print('测试失败')
    #     return False

    # self.status = 1
    # return True
    # 执行d4j自带的测试脚本

# def compile(programDir, self):
#     # 执行d4j自带的编译脚本
#     cmd = ['{}/framework/bin/defects4j'.format(self.configData['D4jHome']), 'compile', '-w', programDir]
    
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     try:
#         # 等待子进程完成，设置超时时间
#         stdout, stderr = process.communicate(timeout=120)

#     except subprocess.TimeoutExpired:
#         # 如果超时，终止子进程
#         process.terminate()
#         process.kill()
        
#         self.status = 0
#         print('编译超时，杀死进程')
#         return False

#     if process.returncode != 0:
#         # 如果子进程返回非零代码，表示编译失败
#         self.status = 0
#         print('编译失败')
#         return False

    # return True

def compile(programDir, self):
    cmd = ['{}/framework/bin/defects4j'.format(self.configData['D4jHome']), 'compile', '-w', programDir]
    
    # 使用preexec_fn设置子进程组
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        # 使用os.killpg发送信号到子进程组
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        self.status = 0
        print('编译超时，杀死进程')
        return False
    except Exception as e:
        # 处理其他可能的异常
        print(f"发生异常: {e}")
        self.status = 0
        return False

    if process.returncode != 0:
        self.status = 0
        print('编译失败')
        return False

    return True

def test(programDir, self):
    cmd = ['{}/framework/bin/defects4j'.format(self.configData['D4jHome']), 'test', '-w', programDir]
    
    # 使用preexec_fn设置子进程组
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        # 使用os.killpg发送信号到子进程组
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        self.status = 0
        print('测试超时，杀死进程')
        return False
    except Exception as e:
        # 处理其他可能的异常
        print(f"发生异常: {e}")
        self.status = 0
        return False

    if process.returncode != 0:
        self.status = 0
        print('测试失败')
        return False

    self.status = 1
    return True

# def test(programDir, self):
#     cmd = ['{}/framework/bin/defects4j'.format(self.configData['D4jHome']), 'test', '-w', programDir]
    
#     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     try:
#         # 等待子进程完成，设置超时时间
#         stdout, stderr = process.communicate(timeout=120)

#     except subprocess.TimeoutExpired:
#         # 如果超时，终止子进程
#         process.terminate()
#         process.kill()
        
#         self.status = 0
#         print('测试超时，杀死进程')
#         return False

#     if process.returncode != 0:
#         # 如果子进程返回非零代码，表示测试失败
#         self.status = 0
#         print('测试失败')
#         return False

#     self.status = 1
#     return True


def checkAndCreateDir(Path):
    if not os.path.exists(Path):
        os.makedirs(Path)

def checkAndCleanDir(Path):
    try:
        shutil.rmtree(Path)
    except OSError as e:
        print("Error: %s : %s" % (Path, e.strerror))
    os.mkdir(Path)

def getTargetStatement(sed_cmd):
    out, err = subprocess.Popen(sed_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    return out, err

def decodeOut(decodeStatement):
    return decodeStatement.decode().strip()

def is_overlap(start1, end1, start2, end2):
    # 判断两个区间是否有交集
    if start1 < end2 and start2 < end1:
        return True
    else:
        return False


def diff_strings(s1, s2):
    """
    计算两个字符串之间的差异并返回不同的子串

    Args:
        s1 (str): 第一个字符串
        s2 (str): 第二个字符串

    Returns:
        list of tuples: 不同的子串列表，每个元组包含四个元素：原始子串、原始子串在 s1 中的起始位置、原始子串在 s1 中的结束位置、第二个字符串中的子串以及第二个字符串中的子串在 s2 中的起始位置和结束位置
    """
    try:
        # 使用 difflib 库计算差异
        diff = difflib.SequenceMatcher(None, s1, s2)
        # 查找不同之处的子串
        substrings = []
        s1_pos = 0
        s2_pos = 0
        for op, i1, i2, j1, j2 in diff.get_opcodes():
            if op == 'replace':
                s1_substring = s1[i1:i2]
                s2_substring = s2[j1:j2]
                substrings.append((op, s1_substring, i1, i2, s2_substring))
            elif op == 'delete':
                s1_substring = s1[i1:i2]
                substrings.append((op, s1_substring, i1, i2, ""))
            elif op == 'insert':
                s2_substring = s2[j1:j2]
                substrings.append((op, s2_substring, i1, i1, s2_substring))
                s1_pos -= len(s2_substring)
            else:
                pass
            s1_pos += i2 - i1
            s2_pos += j2 - j1

        # 将不同之处的子串作为元组列表返回
        return substrings
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')



# 根据修改结果判断能否合并
def merge_strings(origin_mutant1_diff, origin_mutant2_diff, origin_line): 
    try:
        # 遍历差异的操作列表
        for op1, origin1, i1_1, i1_2, mutant1 in origin_mutant1_diff:
            for op2, origin2, i2_1, i2_2, mutant2 in origin_mutant2_diff:
                if op1 == op2 == "replace":
                    if not is_overlap(i1_1, i1_2, i2_1, i2_2):
                        if i1_1 < i2_1:
                            origin_line = origin_line[:i1_1] + mutant1 + origin_line[i1_2:i2_1] + mutant2 + origin_line[i2_2:]
                        else:
                            origin_line = origin_line[:i2_1] + mutant2 + origin_line[i2_2:i1_1] + mutant1 + origin_line[i1_2:]
                        # print(op1, op2, origin_line)
                        return origin_line
                    else: 
                        # print(op1, op2, 'Error')
                        return False
                elif op1 == op2 == "delete":
                    if not is_overlap(i1_1, i1_2, i2_1, i2_2):
                        if i1_1 < i2_1:
                            origin_line = origin_line[:i1_1] + origin_line[i1_2:i2_1] + origin_line[i2_2:]
                        else:
                            origin_line = origin_line[:i2_1] + origin_line[i2_2:i1_1] + origin_line[i1_2:]
                        # print(op1, op2, origin_line)
                        return origin_line
                    else:
                        # print(op1, op2, 'Error')
                        return False
                elif op1 == op2 == "insert":
                    if not is_overlap(i1_1, i1_2, i2_1, i2_2):
                        if i1_1 < i2_1:
                            origin_line = origin_line[:i1_1] + mutant1 + origin_line[i1_1:i2_1] + mutant2 + origin_line[i2_1:]
                        else:
                            origin_line = origin_line[:i2_1] + mutant1 + origin_line[i2_1:i1_1] + mutant2 + origin_line[i1_1:]
                        # print(op1, op2, origin_line)
                        return origin_line
                    else:
                        # print(op1, op2, 'Error')
                        return False
                elif op1 == "replace" and op2 == "delete":
                    if not is_overlap(i1_1, i1_2, i2_1, i2_2):
                        if i1_1 < i2_1:
                            origin_line = origin_line[:i1_1] + mutant1 + origin_line[i1_2:i2_1] + origin_line[i2_2:]
                        else:
                            origin_line = origin_line[:i2_1] + origin_line[i2_2:i1_1] + mutant1 + origin_line[i1_2:]
                        # print(op1, op2, origin_line)
                        return origin_line
                    else:
                        # print(op1, op2, 'Error')
                        return False
                elif op1 == "delete" and op2 == "replace":
                    if not is_overlap(i1_1, i1_2, i2_1, i2_2):
                        if i1_1 < i2_1:
                            origin_line = origin_line[:i1_1] + origin_line[i1_2:i2_1] + mutant2 + origin_line[i2_2:]
                        else:
                            origin_line = origin_line[:i2_1] + mutant2 + origin_line[i2_2:i1_1] + origin_line[i1_2:]
                        # print(op1, op2, origin_line)
                        return origin_line
                    else:
                        # print(op1, op2, 'Error')
                        return False
                elif op1 == "replace" and op2 == "insert":
                    if not is_overlap(i1_1, i1_2, i2_1, i2_2):
                        if i1_1 < i2_1:
                            origin_line = origin_line[:i1_1] + mutant1 + origin_line[i1_2:i2_1] + mutant2 + origin_line[i2_2:]
                        else:
                            origin_line = origin_line[:i2_1] + mutant2 + origin_line[i2_2:i1_1] + mutant1 + origin_line[i1_2:]
                        # print(op1, op2, origin_line)
                        return origin_line
                    else:
                        # print(op1, op2, 'Error')
                        return False
                elif op1 == "insert" and op2 == "replace":
                    if not is_overlap(i1_1, i1_2, i2_1, i2_2):
                        if i1_1 < i2_1:
                            origin_line = origin_line[:i1_1] + mutant1 + origin_line[i1_2:i2_1] + mutant2 + origin_line[i2_2:]
                        else:
                            origin_line = origin_line[:i2_1] + mutant2 + origin_line[i2_2:i1_1] + mutant1 + origin_line[i1_2:]
                        # print(op1, op2, origin_line)
                        return origin_line
                    else:
                        # print(op1, op2, 'Error')
                        return False
                elif op1 == "delete" and op2 == "insert":
                    if not is_overlap(i1_1, i1_2, i2_1, i2_2):
                        if i1_1 < i2_1:
                            origin_line = origin_line[:i1_1] + origin_line[i1_2:i2_1] + mutant2 + origin_line[i2_2:]
                        else:
                            origin_line = origin_line[:i2_1] + mutant2 + origin_line[i2_2:i1_1] + origin_line[i1_2:]
                        # print(op1, op2, origin_line)
                        return origin_line
                    else:
                        # print(op1, op2, 'Error')
                        return False
                elif op1 == "insert" and op2 == "delete":
                    if not is_overlap(i1_1, i1_2, i2_1, i2_2):
                        if i1_1 < i2_1:
                            origin_line = origin_line[:i1_1] + mutant1 + origin_line[i1_2:i2_1] + origin_line[i2_2:]
                        else:
                            origin_line = origin_line[:i2_1] + origin_line[i2_2:i1_1] + mutant1 + origin_line[i1_2:]
                        # print(op1, op2, origin_line)
                        return origin_line
                    else:
                        # print(op1, op2, 'Error')
                        return False
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')



# 判断replace的内容
def changeDiff(s, t):
    try:
        lens = len(s)
        lent = len(t)
        if s[0] == t[0] and s[lens - 1] == t[lent - 1]:
            i1 = -1
            for index, item in enumerate(s):
                if s[index] != t[index]:
                    i1 = index
                    break
            tmpindex = -1
            while s[tmpindex] == t[tmpindex]:
                tmpindex -= 1
            # print(s[i1:tmpindex + 1], i1, tmpindex, t[i1:tmpindex + 1])
            return [('replace', s[i1:tmpindex + 1], i1, lens + tmpindex + 1, t[i1:tmpindex + 1])]
        elif s[0] == t[0]:
            i1 = -1
            for index, item in enumerate(s):
                if s[index] != t[index]:
                    i1 = index
                    break
            return [('replace', s[i1:-1], i1, lens, t[i1:-1])]
        elif s[lens - 1] == t[lent - 1]:
            tmpindex = -1
            i1 = 0
            while s[tmpindex] == t[tmpindex]:
                tmpindex -= 1
            return [('replace', s[i1:tmpindex + 1], i1, lens + tmpindex + 1, t[i1:tmpindex + 1])]
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')


def merge_mutations(original_file, mutated_file1, mutated_file2, line_num1, line_num2):
    try:
        # compare line numbers
        if line_num1 != line_num2:
            line = subprocess.check_output(f"sed -n '{line_num1}p' {mutated_file1}", shell=True).decode().strip()
            for line_num, line_content in enumerate(fileinput.input(original_file, inplace=True), start=1):
                if line_num == line_num1:
                    print(line)
                else:
                    print(line_content, end="")
            line = subprocess.check_output(f"sed -n '{line_num2}p' {mutated_file2}", shell=True).decode().strip()
            for line_num, line_content in enumerate(fileinput.input(original_file, inplace=True), start=1):
                if line_num == line_num2:
                    print(line)
                else:
                    print(line_content, end="")
            # return 
            return True
        else:
            # 使用 sed 命令输出指定行
            origin_sed_cmd = f"sed -n '{line_num1}p' {original_file}"
            mutant1_sed_cmd = f"sed -n '{line_num1}p' {mutated_file1}"
            mutant2_sed_cmd = f"sed -n '{line_num1}p' {mutated_file2}"

            # 调用 sed 命令并获取输出
            origin_statement_out, origin_statement_err = getTargetStatement(origin_sed_cmd)
            mutant1_statement_out, mutant1_statement_err = getTargetStatement(mutant1_sed_cmd)
            mutant2_statement_out, mutant2_statement_err = getTargetStatement(mutant2_sed_cmd)

            # 输出结果
            if  mutant1_statement_out and mutant2_statement_out:
                origin_line = decodeOut(origin_statement_out)
                mutant1_line = decodeOut(mutant1_statement_out)
                mutant2_line = decodeOut(mutant2_statement_out)
                origin_mutant1_diff = diff_strings(origin_line, mutant1_line)
                origin_mutant2_diff = diff_strings(origin_line, mutant2_line)
                if len(origin_mutant1_diff) > 1:
                    origin_mutant1_diff = changeDiff(origin_line, mutant1_line)
                if len(origin_mutant2_diff) > 1:
                    origin_mutant2_diff = changeDiff(origin_line, mutant2_line)
                merge_result = merge_strings(origin_mutant1_diff, origin_mutant2_diff, origin_line)
                print(origin_mutant1_diff, origin_mutant2_diff)
                if merge_result:
                    for line_num, line_content in enumerate(fileinput.input(original_file, inplace=True), start=1):
                        if line_num == line_num1:
                            print(merge_result)
                        else:
                            print(line_content, end="")
                    return True
                else:
                    return False
            else:
                return False
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        print(f'\033[1;31mError at line {line_number}: {e}\033[0m')

        

def copy_file(src_path, dst_path):
    """
    将原文件复制到指定路径，如果目的路径不存在则自动创建
    :param src_path: 原文件路径
    :param dst_path: 目标路径
    :return: 无返回值
    """
    # 创建目标文件夹（如果不存在）
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # 复制文件
    shutil.copy2(src_path, dst_path)


class secondOrderExecutorTool:
    def __init__(self, project, version, muInfo, configData):
        
        # with open('config.json', 'r') as configFile:
        #     # print('开始读取配置文件')
        #     self.configData = json.load(configFile)
        #     # print('读取配置文件完成')
        self.configData = configData
        # 错误程序失败测试用例结果路径
        self.faliingTestOutputPath = self.configData['faliingTestOutputPath']
        # metalix方法
        self.passList = {
                'type1': [],
                'type2': [],
                'type3': [],
                'type4': []}
        self.killList = {
                'type1': [],
                'type2': [],
                'type3': [],
                'type4': []}
        # MUSE方法
        self.mKillList = {}

        # 完整代码的具体版本路径
        self.djSrcPath = os.path.join(self.configData['djSrcPath'], project, version)
        self.tempSrcPath = self.configData['tempSrcPath']
        
        checkAndCreateDir(self.tempSrcPath)
        
        # 项目名称
        self.project = project
        # 项目id
        self.version = version
        
        # 线程锁
        self.OccupiedVersionMutex = threading.Lock()

        self.innerTempSrcPath = ''
        self.muInfo = muInfo
        self.status = 1

        try:
            self.start_copy()
            if(not self.start_muti()):
                self.status = 0
                self.start_remove()
                return
            self.start_compile_run()
            self.start_remove()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            line_number = exc_tb.tb_lineno
            print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
            self.status = 0
            self.start_remove()

    def start_copy(self):
        # print("-------start copy", self.project, self.version, "-------")
        # try:
        # 随机生成字符串
        uid = str(uuid.uuid4())
        suid = ''.join(uid.split('-'))
        self.innerTempSrcPath = os.path.join(self.tempSrcPath, self.project+"-"+self.version+"-"+suid)
        
        self.OccupiedVersionMutex.acquire()
        shutil.copytree(self.djSrcPath, self.innerTempSrcPath)
        self.OccupiedVersionMutex.release()
        # print("-------copy end-------")

    def start_muti(self):
        try:
            # print("-------mutation build-------")
            self.OccupiedVersionMutex.acquire()
            if self.muInfo['relativePath1'] != self.muInfo['relativePath2']:
                # print(self.muInfo['relativePath1'])
                # print(self.muInfo['relativePath2'])
                shutil.copyfile(self.muInfo['mutFilePath1'], os.path.join(self.innerTempSrcPath,self.muInfo['relativePath1']))
                shutil.copyfile(self.muInfo['mutFilePath2'], os.path.join(self.innerTempSrcPath,self.muInfo['relativePath2']))
                self.OccupiedVersionMutex.release()
            else:
                zuhepath = os.path.join(self.configData['SOMmutantsFile'], self.project, self.version, str(self.muInfo['index']))
                checkAndCreateDir(zuhepath)
                originpath = os.path.join(self.djSrcPath, self.muInfo['relativePath1'])
                zuhepath = os.path.join(zuhepath, os.path.basename(originpath))
                if not os.path.exists(zuhepath):
                    path1 = self.muInfo['mutFilePath1']
                    path2 = self.muInfo['mutFilePath2']
                    command = f"""
/home/fanluxi/jdk-17.0.9/bin/java -cp ./jar/codeMerge-1.0-SNAPSHOT.jar:./jar/javaparser-core-3.24.0.jar:./jar/java-diff-utils-4.12.jar:./jar/commons-cli-1.4.jar org.example.Main -pathOri {originpath} -pathM1 {path1} -pathM2 {path2} -pathOutput {zuhepath}
"""
                    # print(originpath)
                    # print(path1)
                    # print(path2)
                    # print(zuhepath)
                    copyCodeFlag = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if copyCodeFlag.returncode == 0:
                        # print(f"成功")
                        shutil.copyfile(zuhepath, os.path.join(self.innerTempSrcPath,self.muInfo['relativePath1']))
                        self.OccupiedVersionMutex.release()
                        return True
                    else:
                        print(f"失败")
                        print(f"错误信息: {copyCodeFlag.stderr}")
                        self.OccupiedVersionMutex.release()
                        return False
                else:
                    # print("已存在")
                    shutil.copyfile(zuhepath, os.path.join(self.innerTempSrcPath,self.muInfo['relativePath1']))
                    self.OccupiedVersionMutex.release()
                    return True
                    
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            line_number = exc_tb.tb_lineno
            print(f'\033[1;31mError at line {line_number}: {e}\033[0m')
            self.OccupiedVersionMutex.release()
            return False
            

        # print("-------mutation build ebd-------")

    def start_compile_run(self):
        # print("-------start compile run-------")
        if not compile(self.innerTempSrcPath, self):
            return
        
        if not test(self.innerTempSrcPath, self):
            return
        
        try:
            # 所有的测试用例
            allTests = []
            # 变异体的失败测试用例
            faileTests = {
                'type1': [],
                'type2': [],
                'type3': [],
                'type4': []
            }
            # 原始程序的失败测试用例
            originFaileTests = {
                'type1': [],
                'type2': [],
                'type3': [],
                'type4': []
            }
            
            # 获取所有的测试用例
            with open(os.path.join(outputCleanPath, self.project, self.version, "all_tests.txt"), 'r', encoding='utf-8') as f:
                allTests = [line.replace("#","::").strip() for line in f.readlines()]
            
            # 把变异体的执行结果存到FOMResult里面去
            # 之所以djSrc-temp里面没有是因为如果执行顺利那么会执行self.start_remove()，从而导致这个变异体文件被删除
            # checkAndCreateDir(os.path.join(faultlocalizationResultPath, self.project, self.version, "failing_tests"))
            # shutil.copy2(os.path.join(self.innerTempSrcPath, "failing_tests"), os.path.join(faultlocalizationResultPath, self.project, self.version, "failing_tests", str(self.muInfo["index"])))

            # region  获取当前执行变异体的执行结果
            with open(os.path.join(self.innerTempSrcPath, "failing_tests"), 'r', encoding='utf-8') as f:
                lines = f.read()
            lines = lines.split('---')
            lines = [s.strip() for s in lines if s.strip()]
            for s in lines:
                testName = s.split('\n')[0].strip()
                faileTests['type1'].append([testName, s.split('\n')[0]])
                faileTests['type2'].append([testName, s.split('\n')[0] + s.split('\n')[1].split(':')[0]])
                faileTests['type3'].append([testName, s.split('\n')[0] + s.split('\n')[1]])
                faileTests['type4'].append([testName, s])
            # endregion

            # region 获取原始程序的执行结果
            with open(os.path.join(self.faliingTestOutputPath, self.project, self.version, "failing_tests"), 'r', encoding='utf-8') as f:
                lines = f.read()
            lines = lines.split('---')
            lines = [s.strip() for s in lines if s.strip()]
            for s in lines:
                testName = s.split('\n')[0].strip()
                originFaileTests['type1'].append([testName, s.split('\n')[0]])
                originFaileTests['type2'].append([testName, s.split('\n')[0] + s.split('\n')[1].split(':')[0]])
                originFaileTests['type3'].append([testName, s.split('\n')[0] + s.split('\n')[1]])
                originFaileTests['type4'].append([testName, s])
            # endregion

            for t in allTests:
                # passList中，0是通过1是失败 killList中0是存活1是杀死
                for i in range(1, 5):
                    # 四种 MBFL 杀死信息粒度 type1-type4
                    faileTestsList = faileTests[f'type{i}']
                    originFaileTestsList = originFaileTests[f'type{i}']

                    # 当前测试用例是否存在于变异体失败测试用例列表
                    flag_fail = t in [test[0] for test in faileTestsList]
                    # 当前测试用例是否存在于原程序失败测试用例列表
                    flag_origin = t in [test[0] for test in originFaileTestsList]
                    # 区分通过还是失败测试用例是针对原程序的
                    if flag_origin:
                        self.passList.get(f'type{i}').append(1)
                    else:
                        self.passList.get(f'type{i}').append(0)

                    # 获取杀死信息
                    if flag_fail and flag_origin:
                        faileInfo_results = [test for test in faileTestsList if test[0] == t]
                        faileInfo = []
                        for faile_filtered in faileInfo_results:
                            faileInfo.append(faile_filtered[1])

                        originFaileInfo_results = [test for test in originFaileTestsList if test[0] == t]
                        originFaileInfo = []
                        for origin_filtered in originFaileInfo_results:
                            originFaileInfo.append(origin_filtered[1])

                        if faileInfo == originFaileInfo:
                            self.killList.get(f'type{i}').append(0)
                        else:
                            self.killList.get(f'type{i}').append(1)
                            # print(i,' xsq no ',faileInfo,originFaileInfo)
                    elif flag_fail:
                        self.killList.get(f'type{i}').append(1)
                    elif flag_origin:
                        self.killList.get(f'type{i}').append(1)
                    else:
                        self.killList.get(f'type{i}').append(0)
            # # 获取变异体执行结果
            # with open(os.path.join(self.innerTempSrcPath, "failing_tests"), 'r', encoding='utf-8') as f:
            #     lines = f.readlines()
            #     for index, line in enumerate(lines):
            #         if '---' not in line:
            #             continue
            #         testName = line.split('::')[-1].strip()
            #         testInfo = lines[index + 1].strip()
            #         faileTests.append([testName, testInfo])
                    
            # # 获取错误程序失败测试用例信息
            # with open(os.path.join(self.faliingTestOutputPath, self.project, self.version, "failing_tests"), 'r', encoding='utf-8') as f:
            #     lines = f.readlines()
            #     for index, line in enumerate(lines):
            #         if '---' not in line:
            #             continue
            #         testName = line.split('::')[-1].strip()
            #         testInfo = lines[index + 1].strip()
            #         originFaileTests.append([testName, testInfo])
                    
            # # 对比生成杀死矩阵
            # for t in allTests:
            #     # passList中，0是通过1是失败 killList中0是存活1是杀死
            #     if t in list(map(lambda x: x[0],faileTests)):
            #         self.passList.append(1)
            #     else:
            #         self.passList.append(0)
            #     if t in list(map(lambda x: x[0],faileTests)) and t in list(map(lambda x: x[0],originFaileTests)):
            #         # self.mKillList.append(0)
            #         filtered_results = list(filter(lambda x: x[0] == t, faileTests))
            #         if len(filtered_results) > 1:
            #             faileInfo = filtered_results[1]
            #         else:
            #             # 处理没有足够元素的情况
            #             faileInfo = None  # 或者您可以根据需求进行其他处理
            #         filtered_results = list(filter(lambda x: x[0] == t, originFaileTests))
            #         if len(filtered_results) > 1:
            #             originFaileInfo = filtered_results[1]
            #         else:
            #             # 处理没有足够元素的情况
            #             originFaileInfo = None  # 或者您可以根据需求进行其他处理

            #         # faileInfo = filter(lambda x: x[0]==t,faileTests)[1]
            #         # faileInfo = list(filter(lambda x: x[0] == t, faileTests))[1]
            #         # originFaileInfo = filter(lambda x: x[0]==t,originFaileTests)[1]
            #         if faileInfo == originFaileInfo:
            #             self.killList.append(0)
            #         else:
            #             self.killList.append(1)
            #     elif t in list(map(lambda x: x[0],faileTests)):
            #         # self.mKillList.append(1)
            #         self.killList.append(1)
            #     elif t in list(map(lambda x: x[0],originFaileTests)):
            #         # self.mKillList.append(1)
            #         self.killList.append(1)
            #     else:
            #         # self.mKillList.append(0)
            #         self.killList.append(0)
            self.status = 1
        except Exception as e:
            self.status = 0
            exc_type, exc_obj, exc_tb = sys.exc_info()
            line_number = exc_tb.tb_lineno
            file_name = exc_tb.tb_frame.f_code.co_filename
            print(f'\033[1;31mError in {file_name} at line {line_number}: {e}\033[0m')
            return
        # print("-------end compile run-------")

        return

    def start_remove(self):
        # print("-------start remove run-------")
        self.OccupiedVersionMutex.acquire()
        shutil.rmtree(self.innerTempSrcPath)
        self.OccupiedVersionMutex.release()
        # print("-------end remove run-------")
        return
