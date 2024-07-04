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
    with open("./mutilFaultVersion.json", "r") as f:
        mutilFaultVersion = json.load(f)
    count = 0
    for projectDir in project.keys():
        # projectDir = "Chart"
        for versionNum in range(1, project[projectDir] + 1):
            versionDir = str(versionNum) + 'b'
            # versionDir = "1b"
            # if failVersion.get(projectDir) and versionDir in failVersion[projectDir]:
            if mutilFaultVersion.get(projectDir) is None or versionDir not in mutilFaultVersion[projectDir]:
                # if os.path.exists(f"/home/fanluxi/pmbfl/SOMfaultlocalizationResult/{projectDir}/{versionDir}"):
                #     print(f"/home/fanluxi/pmbfl/SOMfaultlocalizationResult/{projectDir}/{versionDir}")
                #     shutil.rmtree(f"/home/fanluxi/pmbfl/SOMfaultlocalizationResult/{projectDir}/{versionDir}")
                if os.path.exists(f"/home/fanluxi/pmbfl/SOMfaultlocalizationResult/{projectDir}/{versionDir}/susStatement"):
                    print(f"/home/fanluxi/pmbfl/SOMfaultlocalizationResult/{projectDir}/{versionDir}/susStatement")
                    shutil.rmtree(f"/home/fanluxi/pmbfl/SOMfaultlocalizationResult/{projectDir}/{versionDir}/susStatement")
                if os.path.exists(f"/home/fanluxi/pmbfl/SOMfaultlocalizationResult/{projectDir}/{versionDir}/susFunction"):
                    print(f"/home/fanluxi/pmbfl/SOMfaultlocalizationResult/{projectDir}/{versionDir}/susFunction")
                    shutil.rmtree(f"/home/fanluxi/pmbfl/SOMfaultlocalizationResult/{projectDir}/{versionDir}/susFunction")
            else:
                count+=1
    print(count)