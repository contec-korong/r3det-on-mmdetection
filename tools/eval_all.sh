#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
SET=$(seq 1 50)
val1="python tools/test.py work_dirs/r3det_r50_fpn_2x_768/r3det_r50_fpn_2x_768.py work_dirs/r3det_r50_fpn_2x_768_3aug/epoch_"
val2=".pth --eval mAP"
for i in $SET
do
    echo "${i} is evaluation..."
    val3="${val1}${i}${val2}"
    $val3
done