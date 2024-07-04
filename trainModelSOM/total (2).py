import json
import os

# 定义每个项目的特定版本
projectList = {
    "Chart": ["2b", "7b", "15b", "16b", "21b", "22b", "25b", "26b"],
    "Cli": ["1b", "2b", "9b", "18b", "19b", "20b", "22b", "23b", "24b", "26b", "27b", "29b", "32b", "3b"],
    "Time": ["3b","5b","6b","7b","8b","9b","13b","17b","18b","20b","23b","27b"],
    "Lang": ["1b","3b","4b","5b","7b","8b","10b","12b","13b","15b","17b","18b","19b","27b","31b","32b","34b","35b","36b","41b","42b","46b","53b","58b","60b","62b","63b"],
    "Closure": ["3b", "4b", "6b", "9b", "11b", "13b", "16b", "21b", "22b", "23b", "24b", "25b", "26b", "27b", "32b", "35b", "53b", "64b", "74b", "75b", "76b", "115b"],
    "Math": ["7b", "8b", "18b", "20b", "21b", "23b", "24b", "26b", "28b", "29b", "31b", "35b", "37b", "38b", "40b", "43b", "46b", "47b", "49b", "50b", "52b", "56b", "60b", "61b", "62b", "65b", "66b", "67b", "68b", "72b", "76b", "79b", "81b", "83b", "86b", "87b", "88b", "91b", "92b", "93b", "95b", "97b", "100b", "101b", "102b", "103b"],
    "Codec": ["6b"],
    "Compress": ["2b","3b","5b","6b","7b","8b"],
    "Csv": ["8b"]
}

if __name__ == "__main__":
    # 加载包含无效版本信息的JSON文件
    with open("../mutationTool/failVersion.json", "r") as f:
        failVersion = json.load(f)

    for project, versions in projectList.items():
        LOC_num = 0
        Tests_num = 0
        Faults_num = 0

        for version in versions:
            # 检查是否应该跳过当前版本
            if failVersion.get(project) and version in failVersion[project]:
                continue

            try:
                # 统计代码总行数
                with open(f"../../d4j/outputClean/{project}/{version}/hugeCode.txt") as f:
                    LOC_num += sum(1 for _ in f)
                # 统计测试用例数
                with open(f"../../d4j/outputClean/{project}/{version}/all_tests.txt") as f:
                    Tests_num += sum(1 for _ in f)
                # 统计故障数
                with open(f"../../d4j/outputClean/{project}/{version}/faultHuge.txt", 'r') as f:
                    data = json.load(f)
                    for key in data:
                        Faults_num += len(data[key])
            except FileNotFoundError as e:
                print(f"File not found for {project} {version}: {e}")

        print(f"{project}: Versions Counted: {len(versions)}, LOC: {LOC_num}, Tests: {Tests_num}, Faults: {Faults_num}")
