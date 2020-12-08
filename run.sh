#!/bin/sh

mkdir -p /tmp/emnlp2017
curl https://storage.googleapis.com/deepmind-scratchgan-data/train.json --output /tmp/emnlp2017/train.json
curl https://storage.googleapis.com/deepmind-scratchgan-data/valid.json --output /tmp/emnlp2017/valid.json
curl https://storage.googleapis.com/deepmind-scratchgan-data/test.json --output /tmp/emnlp2017/test.json
curl https://storage.googleapis.com/deepmind-scratchgan-data/glove_emnlp2017.txt --output /tmp/emnlp2017/glove_emnlp2017.txt

yum install python3 

pip3 install -r requirements.txt

python3 -m train.py
