import json
import os
import shutil

project = {
    "Chart": 26,
    "Cli": 39,
    "Closure": 133,
    "Closure": 176,
    "Codec": 18,
    "Collections": 4, # 无文件
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
    "Time": 27
}

if __name__ == '__main__':
    with open("./failVersion.json", "r") as f:
        failVersion = json.load(f)
    count = 0
    total = 0
    mutilFaultVersion = {}
    # 遍历源目录
    for root, dirs, files in os.walk("/home/fanluxi/pmbfl/SOMfaultlocalizationResult"):
        for file in files:
            # 拼接源文件路径
            source_file_path = os.path.join(root, file)
            if file.endswith("faultLocalization.json"):
                total += 1
                with open(source_file_path, "r") as f:
                    data = json.load(f)
                if len(data) == 1 and len(list(data.values())[0]) != 1:
                    count += 1
                    project = source_file_path.split("/")[5]
                    version = source_file_path.split("/")[6]
                    if mutilFaultVersion.get(project) is None:
                        mutilFaultVersion[project] = []
                    mutilFaultVersion[project].append(version)
                else:
                    print(source_file_path)
    with open("./mutilFaultVersion.json", "w") as f:
        f.write(json.dumps(mutilFaultVersion, indent=2))
    print(count)
    print(total)