import json
import os
import operator
    
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
def calMcbfl(path1):
    try:
        with open(path1, 'r') as f:
            mbflSus = json.load(f)
        mcbflSus = dict()
        for line in sbflSus.keys():
            if mcbflSus.get(line) is None:
                mcbflSus[line] = dict()
            for method in sbflSus[line].keys():
                if mcbflSus[line].get(method) is None:
                    mcbflSus[line][method] = dict()
                for key in sbflSus[line][method].keys():
                    if mcbflSus[line][method].get(key) is None:
                        mcbflSus[line][method][key] = sbflSus[line][method][key]
                    else:
                        mcbflSus[line][method][key] += sbflSus[line][method][key]
        for line in mbflSus.keys():
            if mcbflSus.get(line) is None:
                mcbflSus[line] = dict()
            for method in mbflSus[line].keys():
                if mcbflSus[line].get(method) is None:
                    mcbflSus[line][method] = dict()
                for key in mbflSus[line][method].keys():
                    if mcbflSus[line][method].get(key) is None:
                        mcbflSus[line][method][key] = mbflSus[line][method][key]
                    else:
                        mcbflSus[line][method][key] += mbflSus[line][method][key]
        for line in mcbflSus.keys():
            for method in mcbflSus[line].keys():
                for key in mcbflSus[line][method].keys():
                    mcbflSus[line][method][key] /= 2
        for item in mcbflSus.keys():
            for method in mcbflSus[line].keys():
                mcbflSus[item][method] = dict(sorted(mcbflSus[item][method].items(),
                                                                        key=operator.itemgetter(1), reverse=True))
        with open(path1, 'w') as f:
            f.write(json.dumps(mcbflSus, indent=2))
    except Exception as e:
        print(e)

if __name__ =="__main__":
    faultlocalizationResultPath = "/home/fanluxi/pmbfl/faultlocalizationResult"
    count = 0
    with open("../mutationTool/failVersion.json", "r") as f:
        failVersion = json.load(f)
    for project in projectList.keys():
        # project = "Time"
        for versionNum in range(1, projectList[project] + 1):
            version = str(versionNum) + 'b'
            if not failVersion.get(project) is None and version in failVersion[project]:
                continue
            susPath = os.path.join(faultlocalizationResultPath, project, version, "sus")
            try:
                with open(susPath + '/sbfl.json', 'r') as f:
                    sbflSus = json.load(f)
                with open(susPath + '/complete.json', 'r') as f:
                    mbflSus = json.load(f)
                mcbflSus = dict()
                for line in sbflSus.keys():
                    if mcbflSus.get(line) is None:
                        mcbflSus[line] = dict()
                    for method in sbflSus[line].keys():
                        if mcbflSus[line].get(method) is None:
                            mcbflSus[line][method] = dict()
                        for key in sbflSus[line][method].keys():
                            if mcbflSus[line][method].get(key) is None:
                                mcbflSus[line][method][key] = sbflSus[line][method][key]
                            else:
                                mcbflSus[line][method][key] += sbflSus[line][method][key]
                for line in mbflSus.keys():
                    if mcbflSus.get(line) is None:
                        mcbflSus[line] = dict()
                    for method in mbflSus[line].keys():
                        if mcbflSus[line].get(method) is None:
                            mcbflSus[line][method] = dict()
                        for key in mbflSus[line][method].keys():
                            if mcbflSus[line][method].get(key) is None:
                                mcbflSus[line][method][key] = mbflSus[line][method][key]
                            else:
                                mcbflSus[line][method][key] += mbflSus[line][method][key]
                for line in mcbflSus.keys():
                    for method in mcbflSus[line].keys():
                        for key in mcbflSus[line][method].keys():
                            mcbflSus[line][method][key] /= 2
                for item in mcbflSus.keys():
                    for method in mcbflSus[line].keys():
                        mcbflSus[item][method] = dict(sorted(mcbflSus[item][method].items(),
                                                                                key=operator.itemgetter(1), reverse=True))
                with open(susPath + '/mcbfl.json', 'w') as f:
                    f.write(json.dumps(mcbflSus, indent=2))
                calMcbfl(susPath + '/complete_based_sbfl.json')
                calMcbfl(susPath + '/hmer.json')
                calMcbfl(susPath + '/knn_based_sbfl.json')
                calMcbfl(susPath + '/knn.json')
                calMcbfl(susPath + '/lr_based_sbfl.json')
                calMcbfl(susPath + '/lr.json')
                calMcbfl(susPath + '/mlp_based_sbfl.json')
                calMcbfl(susPath + '/mlp.json')
                calMcbfl(susPath + '/nb_based_sbfl.json')
                calMcbfl(susPath + '/nb.json')
                calMcbfl(susPath + '/rf_based_sbfl.json')
                calMcbfl(susPath + '/rf.json')
                calMcbfl(susPath + '/pmt.json')
            except Exception as e:
                print(e)
            # exit(1)