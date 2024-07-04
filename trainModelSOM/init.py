import os
import shutil
projectList = {
    "Chart": 26,
    "Cli": 39,
    "Closure": 176,
    "Codec": 18,
    # "Collections": 4,
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
def delete_files_and_folders(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        # 如果是文件夹，递归删除其中的文件和文件夹
        if os.path.isdir(file_path):
            delete_files_and_folders(file_path)
            # 删除空文件夹
            if not os.listdir(file_path):
                os.rmdir(file_path)
        # 如果是文件，删除除了指定文件外的所有文件
        else:
            if file_name not in ["faultLocalization.json", "muInfo.json", "muResult.json", "suspiciousSbfl.json"]:
                os.remove(file_path)

# 调用函数删除指定文件夹下的文件和文件夹
for project in projectList.keys():
    # project = "Cli"
    for versionNum in range(1, projectList[project] + 1):
        try:
            delete_files_and_folders("/home/fanluxi/pmbfl/faultlocalizationResult/" + project + "/" + str(versionNum) + 'b')
        except:
            continue
