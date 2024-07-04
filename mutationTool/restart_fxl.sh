#!/bin/bash

# 杀死fanluxi用户下所有的python和java程序
pkill -u fanluxi python
pkill -u fanluxi java

# 切换目录
cd /home/fanluxi/pmbfl/mutationTool/tool
chmod 777 runMajor.sh
cd ..

# 启动main.py
nohup python3 main.py &
