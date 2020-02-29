#!/bin/bash

dataset_src='/mnt/btrfs-data/dataset/dynamic-hand-gestures/LMDHG'

for i in "$dataset_src"/*.txt
do
	echo $i
	./dynamic-hand-gestures.py $i --view-name top --dataset-path ../dataset-csv/ --no-show-label
done
