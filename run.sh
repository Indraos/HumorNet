#!/bin/sh

mkdir data
cd data
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
cd ..

yum install gcc gcc-c++

pip3 install -r requirements.txt

jupyter notebook --no-browser port=8888
