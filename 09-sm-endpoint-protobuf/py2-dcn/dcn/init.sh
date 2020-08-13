#!/bin/bash

mkdir -p ./dcn/data
mkdir -p ./dcn/output
mkdir -p ./dcn/external/mxnet
mkdir -p ./dcn/model/pretrained_model

cd ./dcn/lib/bbox
python setup_linux.py build_ext --inplace
cd ..
cd ./dataset/pycocotools
python setup_linux.py build_ext --inplace
cd ../..
cd ./nms
python setup_linux.py build_ext --inplace
cd ../..