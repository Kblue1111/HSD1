import json
import csv
import os
column_names = ['dstar', 'ochiai', 'gp13', 'op2', 'jaccard', 'russell', 'turantula', 'naish1', 'binary', 'crosstab', 'muse']
file_names = ['complete', 'complete_based_sbfl', 'sbfl', 'pmt', 'hmer', 'knn_based_sbfl', 
                'knn', 'lr_based_sbfl', 'lr',
                'mlp_based_sbfl', 'mlp', 'nb_based_sbfl', 
                'nb', 'rf_based_sbfl', 'rf', 'mcbfl'
                ]
metr = ['top1', 'top2', 'top3', 'top4', 'top5', 'top10']
model_names = ['rfModel', 'rfModel_based_sbfl', 'knnModel', 'knnModel_based_sbfl', 'lrModel', 'lrModel_based_sbfl',
               'mlpModel', 'mlpModel_based_sbfl', 'nbModel', 'nbModel_based_sbfl']

def exportMAR(rootdir):
    metrics = 'MAR'
    result = dict()
    with open("./result/MAR.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        metricsPath = os.path.join(rootdir, metrics)
        for root, dirs, files in os.walk(metricsPath):
            for file in files:
                if file == 'JxPath' or file == 'Mockito':
                    continue
                with open(metricsPath + "/" + file, 'r') as f:
                    data = json.load(f)
                for filename, item in data.items():
                    for file_name in file_names:
                        if result.get(file_name) == None:
                            result[file_name] = dict()
                        if filename == 'MAR_muse.json':
                            for col in column_names:
                                if item.get(col) != None:
                                    if result['first_order'].get(file) == None:
                                        result['first_order'][file] = dict()
                                    result['first_order'][file][col] = item[col]
                            
                        if file_name in filename:
                            if 'sbfl' not in file_name and 'sbfl' in filename:
                                continue
                            for col in column_names:
                                if item.get(col) != None:
                                    if result[file_name].get(file) == None:
                                        result[file_name][file] = dict()
                                    result[file_name][file][col] = item[col]
        print(result)
        for file_name in file_names:
            # 获取所有键（列名）
            header = ["Method"] + list(result[file_name].keys())

            # 获取所有行数据
            rows = []
            # print(file_name, result[file_name])
            for method, values in result[file_name]['Chart'].items():
                row = [method] + [d[method] for d in result[file_name].values() if d.get(method) != None]
                rows.append(row)
            writer.writerow([file_name])
            writer.writerow(header)
            writer.writerows(rows)

def exportMFR(rootdir):
    metrics = 'MFR'
    result = dict()
    with open("./result/MFR.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        metricsPath = os.path.join(rootdir, metrics)
        for root, dirs, files in os.walk(metricsPath):
            for file in files:
                if file == 'JxPath' or file == 'Mockito':
                    continue
                with open(metricsPath + "/" + file, 'r') as f:
                    data = json.load(f)
                for filename, item in data.items():
                    for file_name in file_names:
                        if result.get(file_name) == None:
                            result[file_name] = dict()
                        if filename == 'MFR_muse.json':
                            for col in column_names:
                                if item.get(col) != None:
                                    if result['first_order'].get(file) == None:
                                        result['first_order'][file] = dict()
                                    result['first_order'][file][col] = item[col]
                        if file_name in filename:
                            # if 'sbfl' not in file_name and 'sbfl' in filename:
                            #     continue
                            for col in column_names:
                                if item.get(col) != None:
                                    if result[file_name].get(file) == None:
                                        result[file_name][file] = dict()
                                    result[file_name][file][col] = item[col]
        # print(result)
        for file_name in file_names:
            # 获取所有键（列名）
            header = ["Method"] + list(result[file_name].keys())

            # 获取所有行数据
            rows = []
            # print(file_name, result[file_name])
            for method, values in result[file_name]['Chart'].items():
                row = [method] + [d[method] for d in result[file_name].values() if d.get(method) != None]
                rows.append(row)
            writer.writerow([file_name])
            writer.writerow(header)
            writer.writerows(rows)


def exportModelMetrics(rootdir):
    metrics = 'ModelMetrics'
    result = dict()
    with open("./result/ModelMetrics.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        metricsPath = os.path.join(rootdir, metrics)
        for root, dirs, files in os.walk(metricsPath):
            for file in files:
                if file == 'JxPath' or file == 'Mockito':
                    continue
                with open(metricsPath + "/" + file, 'r') as f:
                    data = json.load(f)
                for filename, item in data.items():
                    if result.get(file) ==None:
                        result[file] = dict()
                    for file_name in model_names:
                        if result.get(file_name) == None:
                            result[file_name] = dict()
                        if file_name in filename:
                            if 'sbfl' not in file_name and 'sbfl' in filename:
                                continue
                            if result[file_name].get(file) == None:
                                result[file_name][file] = dict()
                            # print(filename)
                            if 'musekill' in filename:
                                result[file_name][file]['muse'] = item
                            elif 'kill' in filename:
                                result[file_name][file]['meta'] = item
                                
        # print(result)
        for file_name in model_names:
            
            # 获取所有键（列名）
            header = ["Project", "Method"] + list(result[file_name]["Chart"]["meta"].keys())

            # 获取所有行数据
            rows = []
            for category, methods in result[file_name].items():
                for method, values in methods.items():
                    row = [category, method] + list(values.values())
                    rows.append(row)
            writer.writerow([file_name])
            writer.writerow(header)
            writer.writerows(rows)

def exportTopN(rootdir):
    topNpath = os.path.join(rootdir, "topN")
    result = dict()
    with open("./result/topN.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        for root, dirs, files in os.walk(topNpath):
            for file in files:
                if file == 'JxPath' or file == 'Mockito':
                    continue
                with open(topNpath + "/" + file, 'r') as f:
                    data = json.load(f)
                for filename, item in data.items():
                    for file_name in file_names:
                        if result.get(file_name) == None:
                            result[file_name] = dict()
                        if file_name + ".json" == filename:
                            for col in column_names:
                                if item.get(col) != None:
                                    if result[file_name].get(col) == None:
                                        result[file_name][col] = dict()
                                    for me in metr:
                                        if result[file_name][col].get(me) == None:
                                            result[file_name][col][me] = item[col][me]
                                        else:
                                            result[file_name][col][me] += item[col][me]
                                            
        print(result)
        for file_name in file_names:
            # 获取所有键（列名）
            header = ["Method"]

            # 获取所有行数据
            rows = []
            print(file_name)
            for method in result[file_name].keys():
                row = [method]
                # for category in result[file_name].keys():
                #     if result[file_name][category].get(method) == None:
                #         continue
                row.extend([result[file_name][method]["top1"], result[file_name][method]["top2"], result[file_name][method]["top3"], result[file_name][method]["top4"], result[file_name][method]["top5"], result[file_name][method]["top10"]])
                rows.append(row)
            writer.writerow([file_name])
            writer.writerow(header)
            writer.writerows(rows)


if __name__ =="__main__":
    rootdir = "/home/fanluxi/pmbfl/trainModel"
    exportMAR(rootdir)
    exportMFR(rootdir)
    # exportModelMetrics(rootdir)
    exportTopN(rootdir)