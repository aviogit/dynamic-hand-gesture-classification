#!/bin/bash

echo 'Creating dataset directory...'

lastdatasetno=0
lastdataset=`ls dataset-v* -d 2>/dev/null | sort | tail -1`
lastdatasetno=${lastdataset##dataset-v}

this_datasetno=$((lastdatasetno+1))

mkdir -p dataset-v$this_datasetno
cd dataset-v$this_datasetno

echo "Created dataset directory dataset-v$this_datasetno"
pwd
cd ..

for i in 2020*
do
	cd $i
	pwd
	#echo "About to copy PNG files in 10 seconds..."
	#sleep 10
	echo "About to copy PNG files"
	rsync *.png ../dataset-v$this_datasetno/$i
	cd ../dataset-v$this_datasetno/$i
	mmv '*.png' '#1-'"$i"'.png'
	mv *.png ..
	cd ..
	rmdir $i
	cd ..
done

cd dataset-v$this_datasetno
classes=`ls | sed 's:^\([A-Za-z()-]*\)-[0-9]*-.*.png:\1:g' | sort | uniq`
echo "Creating directories for the following classes: $clean_classes"
for cl in $classes
do
	clean_class=`echo $cl | sed 's:\([a-z-]*\)\([a-z()-]*\):\1:g' | sed -e 's:-$::g'`
	echo "Creating directory $clean_class..."
	mkdir $clean_class
	mv "$cl"* $clean_class
done

exit 0
