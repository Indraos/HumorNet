#!/bin/sh

yum install gcc gcc-c++
pip3 install -r requirements.txt
python3 run generate_and_classify.py