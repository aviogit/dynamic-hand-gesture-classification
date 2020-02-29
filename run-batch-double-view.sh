#!/bin/bash

dataset_src='/mnt/btrfs-data/dataset/dynamic-hand-gestures/LMDHG'
dataset_csv='/mnt/btrfs-data/dataset/dynamic-hand-gestures/dataset-csv'

for i in "$dataset_src"/*.txt
do
	echo $i
	./dynamic-hand-gestures.py $i --view-name top --dataset-path "$dataset_csv" --no-show-label --double-view --batch-mode
done
