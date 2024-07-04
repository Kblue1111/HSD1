import json
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

if __name__ == "__main__":
    
    with open("../mutationTool/failVersion.json", "r") as f:
        failVersion = json.load(f)

    for project in projectList.keys():
        version_num = 0
        LOC_num = 0
        Tests_num = 0
        Faults_num = 0
        for versionNum in range(1, projectList[project] + 1):
            version = str(versionNum) + 'b'
            if not failVersion.get(project) is None and version in failVersion[project]:
                continue
            version_num += 1
            with open("../../d4j/outputClean/" + project + "/" + version + "/hugeCode.txt") as f:
                for line in f:
                    LOC_num += 1
            with open("../../d4j/outputClean/" + project + "/" + version + "/all_tests.txt") as f:
                for line in f:
                    Tests_num += 1
            with open("../../d4j/outputClean/" + project + "/" + version + "/faultHuge.txt", 'r') as f:
                data = json.load(f)
                for key in data.keys():
                    Faults_num += len(data[key])
        print(project, version_num, LOC_num, Tests_num, Faults_num)
            
            
                
            