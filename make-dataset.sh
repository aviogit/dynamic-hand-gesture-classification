#!/bin/bash

echo 'Creating dataset directory...'
mkdir -p dataset
cd dataset
echo 'Moving all the PNG files into dataset directory...'
mv ../*datafile*.png .

# mmv '* * *-*.png' '#1-#2-#3-#4.png'
# mmv '* *-*.png' '#1-#2-#3.png'

echo 'Creating subdirectories and sorting PNG files into proper gesture subdirectory...'
for i in `cat ../info/gesture-classes.txt | awk '{print $2}'`
do
	gest=$i
	echo $gest
	mkdir $gest
	for j in $gest-[0-9]*-datafile*.png
	do
		mv "$j" $gest
	done
done

exit 0

# To split as in the paper (train = 1-35, valid = 36-50) do something like this:
#
# for i in train/* ; do gest=${i##train/} ; src=$i ; echo $i ; dst=valid/$gest ; echo $dst ; mkdir -p $dst ; ls -l $src/*-datafile3[6-9]*.png $dst ; done

