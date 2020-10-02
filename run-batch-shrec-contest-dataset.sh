#!/bin/bash

currdate()
{
 echo $(date +"%Y%m%d")
}
currtime()
{
 echo $(date +"%H%M%S")
}


dataset_src='./shrec-contest-dataset/20200415-new-conversion.py/training-set-converted'

declare -a pids

for hst in 50 100 150 200 250 300 350 400					# 8 processes in parallel
do
	captured_images_output_dir=./`currdate`-`currtime`-"hst-$hst"
	for fname in "$dataset_src"/*.csv.xz
	do
		echo "Processing file $fname with --clear-history-older-than-n-frames $hst"
		#./dynamic-hand-gestures.py $i --view-name top --dataset-path ./shrec-contest-dataset --no-show-label
		#./dynamic-hand-gestures.py "$i" --dataset-path ./shrec-contest-dataset --data-scale-factor 600.0 --data-x-offset -200.0 --data-y-offset -1200.0 --fps 10
		#./sleep-30-seconds.sh $fname												\
		./dynamic-hand-gestures.py $fname											\
			--dataset-path "$dataset_src"											\
			--captured-images-output-dir "$captured_images_output_dir"							\
			--batch-mode --no-show-label											\
			--data-scale-factor 600.0 --data-x-offset -100 --data-y-offset -1250.0 --data-z-offset -250			\
			--clear-history-older-than-n-frames $hst									\
			--screenshot-every-n-frames 25											&

		pid=$!
		pids[$pid]=1
		echo "Launched process with PID: $this_pid -> pids[$pid] = ${pids[$pid]}"

		while (true)
		do
			sleep 0.1
			counter=0
			for this_pid in "${!pids[@]}"; do
				echo -n "$this_pid - "
				is_running=`ps x | grep "^ *$this_pid "`
				#echo $is_running
				if [ ! -z "$is_running" ] ; then		# if grep returns something -> PID is running
					counter=$((counter+1))
				else
					echo "Process pids[$this_pid] has finished."
					unset "pids[$this_pid]"
				fi
			done
			if [ $counter -lt 8 ] ; then
				echo "Counter is $counter, breaking infinite loop and launching another process..."
				break
			fi
		done
	done
done
