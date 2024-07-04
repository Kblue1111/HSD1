from datetime import datetime, timedelta

# 初始化每个模型的总时间
total_time_logisticRegressionModel = timedelta()
total_time_naiveBayesModel = timedelta()
total_time_randomForestModel = timedelta()

# 打开日志文件
with open('pmt.log', 'r') as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'logisticRegressionModel' in line and 'success' not in line:
            # 解析时间字符串并添加到总时间中
            time_str = lines[i+1].strip().split()[-1]
            time_parts = time_str.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = float(time_parts[2])
            total_time_logisticRegressionModel += timedelta(hours=hours, minutes=minutes, seconds=seconds)
        elif 'naiveBayesModel' in line and 'success' not in line:
            time_str = lines[i+1].strip().split()[-1]
            time_parts = time_str.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = float(time_parts[2])
            total_time_naiveBayesModel += timedelta(hours=hours, minutes=minutes, seconds=seconds)
        elif 'randomForestModel' in line and 'success' not in line:
            time_str = lines[i+1].strip().split()[-1]
            time_parts = time_str.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = float(time_parts[2])
            total_time_randomForestModel += timedelta(hours=hours, minutes=minutes, seconds=seconds)
        i += 1

# 打印每个模型的总时间
print("Total time for logisticRegressionModel:", total_time_logisticRegressionModel)
print("Total time for naiveBayesModel:", total_time_naiveBayesModel)
print("Total time for randomForestModel:", total_time_randomForestModel)

# 计算总时间的平均值
total_time_logisticRegressionModel /= 136
total_time_naiveBayesModel /= 136
total_time_randomForestModel /= 136

# 打印每个模型的平均时间
print("Average time for logisticRegressionModel:", total_time_logisticRegressionModel)
print("Average time for naiveBayesModel:", total_time_naiveBayesModel)
print("Average time for randomForestModel:", total_time_randomForestModel)