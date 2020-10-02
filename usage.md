# Dynamic Hand Gestures Classifier

Dynamic Hand Gestures Classifier trained on LMDHG dataset: https://www-intuidoc.irisa.fr/en/english-leap-motion-dynamic-hand-gesture-lmdhg-database/

The classifier has also been trained on our new CNR Hand Gestures dataset (2k images with blank and noise classes): https://github.com/aviogit/dynamic-hand-gesture-classification-datasets/tree/master/dynamic-hand-gestures-new-CNR-dataset-2k-images

The classifier has recently been trained on the SHREC 2020 contest dataset: http://www.andreagiachetti.it/shrec20gestures/


Papers:

1. Lupinetti, K, Ranieri, A, Giannini, F, Monti, M., «_3D dynamic hand gestures recognition using the leap motion sensor and convolutional neural networks_». 2020. arXiv:2003.01450.

`@misc{lupinetti20203d,`<br>
`    title={3D dynamic hand gestures recognition using the Leap Motion sensor and convolutional neural networks},`<br>
`    author={Katia Lupinetti and Andrea Ranieri and Franca Giannini and Marina Monti},`<br>
`    year={2020},`<br>
`    eprint={2003.01450},`<br>
`    archivePrefix={arXiv},`<br>
`    primaryClass={cs.CV}`<br>
`}`


# Real-time inference

`#> ./dynamic-hand-gestures.py --enable-ros --model-name models/resnet-50-img_size-540-960-4a-2020-02-25_11.05.03-epoch-8-stronger-data-augmentation-new-cnr-dataset-2k-images.pkl`

`#> ./dynamic-hand-gestures.py /mnt/data/ros/src/hand/dynamic-hand-gestures-datasets/dataset-csv-xz/datafile42.csv.xz --dataset-path /mnt/data/ros/src/hand/dynamic-hand-gestures-datasets/dataset-csv-xz --do-inference --model-name models/resnet-50-img_size-540-960-4a-2020-02-25_11.05.03-epoch-8-stronger-data-augmentation-new-cnr-dataset-2k-images.pkl`

`#> ./dynamic-hand-gestures.py /mnt/data/ros/src/hand/dynamic-hand-gestures-datasets/dataset-csv-xz/datafile42.csv.xz --dataset-path /mnt/data/ros/src/hand/dynamic-hand-gestures-datasets/dataset-csv-xz --do-inference --model-name models/resnet-50-img_size-540-960-4a-2020-02-25_11.05.03-epoch-8-stronger-data-augmentation-new-cnr-dataset-2k-images.pkl --cuda-device cpu`

`#> ./dynamic-hand-gestures.py ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/output-dataset/datafile-tap-31.csv.xz --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/output-dataset/ --do-inference --model-name models/resnet-50-img_size-540-960-4a-2020-04-14_15.48.20-SHREC-contest-dataset-first-attempt-transfer-learning-from-CNR-dataset.pkl --cuda-device cpu --inference-every-n-frames 20 --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250`

`#> ./dynamic-hand-gestures.py ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/sequences-converted/unknown-2.csv.xz --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/sequences-converted/ --do-inference --model-name models/resnet-50-img_size-540-960-4a-2020-04-14_15.48.20-SHREC-contest-dataset-first-attempt-transfer-learning-from-CNR-dataset.pkl --cuda-device cpu --inference-every-n-frames 10 --reset-history-only-when-prob-greater-than 0.85 --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --fps 10 --clear-history-older-than-n-frames 150 --save-image-only-when-prob-greater-than 0.85`

#### It woooorks!

`#> ./dynamic-hand-gestures.py ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/sequences-converted/unknown-4.csv.xz --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/sequences-converted/ --do-inference --model-name models/resnet-50-img_size-540-960-4a-2020-04-21_15.47.18-SHREC-contest-dataset-transfer-learning-from-CNR-dataset-data-augmentation-with-partial-gestures-and-noise.pkl --cuda-device cpu --inference-every-n-frames 20 --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --fps 10 --save-image-only-when-prob-greater-than 0.98`


#### SHREC 2020 Contest

`#> for i in ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/sequences-converted/unknown-*.csv.xz ; do echo "Processing file: $(basename $i) - $i" ; ./dynamic-hand-gestures.py $i --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/sequences-converted/ --do-inference --model-name models/resnet-50-img_size-540-960-4a-2020-04-21_15.47.18-SHREC-contest-dataset-transfer-learning-from-CNR-dataset-data-augmentation-with-partial-gestures-and-noise.pkl --cuda-device cpu --inference-every-n-frames 20 --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --save-image-only-when-prob-greater-than 0.98 --write-csv-predictions ; done`

`#> ll | nl`
`     1	20200424-210835.914746`
`     2	20200424-210926.902746`
`	...
`    71	20200424-220947.169131`
`    72	20200424-221059.898594`

`#> find . -type f -name '*.csv' -exec grep -v 'filename,num detected' {} \; | sort --field-separator=, -k1 -V | sed 's:^0,unknown-\([0-9]*\).csv:\1:g' > /tmp/submission.csv`



# Dataset creation

`#> for i in ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/*.csv.xz ; do ./dynamic-hand-gestures.py $i --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/ --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --fps 10 --clear-history-older-than-n-frames 150 --screenshot-every-n-frames 25 --batch-mode --no-show-label ; done`

`#> for i in ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/*.csv.xz ; do ./dynamic-hand-gestures.py $i --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/ --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --clear-history-older-than-n-frames 200 --screenshot-every-n-frames 25 --batch-mode --no-show-label ; done`

`#> for i in ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/*.csv.xz ; do ./dynamic-hand-gestures.py $i --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/ --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --clear-history-older-than-n-frames 250 --screenshot-every-n-frames 25 --batch-mode --no-show-label ; done`

`#> for i in ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/*.csv.xz ; do ./dynamic-hand-gestures.py $i --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/ --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --clear-history-older-than-n-frames 300 --screenshot-every-n-frames 25 --batch-mode --no-show-label ; done`

`#> for i in ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/*.csv.xz ; do ./dynamic-hand-gestures.py $i --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/ --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --clear-history-older-than-n-frames 350 --screenshot-every-n-frames 25  --batch-mode --no-show-label ; done`

`#> for i in ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/*.csv.xz ; do ./dynamic-hand-gestures.py $i --dataset-path ../dynamic-hand-gestures-datasets/shrec-2020-contest-dataset/20200415-new-conversion.py/training-set-converted/ --data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250 --clear-history-older-than-n-frames 400 --screenshot-every-n-frames 25 --batch-mode --no-show-label ; done`


### Dataset files renaming/moving

`#> for j in variable-history-length* ; do echo $j ; cd $j ; for i in 202004* ; do echo $j-$i ; cd $i ; pwd ; ~/custom-bin/mmv '*.csv.png' '#1-'"$j-$i"'.csv.png' ; cd .. ; done ; cd .. ; done`

`#> for i in 202004* ; do echo $i ; cd $i ; pwd ; ~/custom-bin/mmv '*.csv.png' '#1-'"$i"'.csv.png' ; cd .. ; done`

`#> while read -r line ; do class=\`echo "$line" | sed 's: :-:g' | awk '{print $2}'\` ; clean_class=\`echo $class | sed 's:-(static)::g'\` ; echo "$class - $clean_class" ; mkdir -p $clean_class ; mv $clean_class*.png $clean_class ; done < shrec-2020-contest-dataset-gesture-classes.txt`

##### Dataset composed of image sequences
After having processed the entire SFINGE 3D dataset in batch, rename the directories so that they also contain the label of the first image in the directory in their name:

`#> for i in 20200921-1* ; do label=`ls $i | head -1 | sed 's:\(.*\)\(-\(.*\)\)*-\([0-9]*\)-datafile:\1 :g' | awk '{print $1}' | sed 's:-(static)::g'` ; echo mv $i $i-$label ; mv $i $i-$label ; done`


### Convert from ROSbag to csv.xz file format (rows of 138 float + 1 label)

`#> rosbag play your_leapmotion_gestures_rosbag.bag`
`#> ./dynamic-hand-gestures.py --enable-ros --clear-history-older-than-n-frames 2500 --batch-mode --save-to-file Shaking.csv.xz --label-to-file Shaking --fps 500`

#### Then you can replay the saved file with:

`#> ./dynamic-hand-gestures.py /tmp/dynamic-hand-gestures-datasets/cnr-dataset-csv-xz/Shaking.csv.xz --show-fps --fps 50 --clear-history-older-than-n-frames 2500 --dataset-path /tmp/dynamic-hand-gestures-datasets/cnr-dataset-csv-xz`
