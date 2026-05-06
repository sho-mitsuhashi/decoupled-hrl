#!/bin/bash

trap 'echo "Terminating process group..."; trap - SIGINT SIGTERM; kill -TERM -$$' SIGINT SIGTERM

mkdir trainlogs


gpus=(0)           # GPU list e.g.(0 1 2)
gpu_n=${#gpus[@]}        # GPU num
idx=0                    




MAX_JOBS=3 



USE_CAPABLE=true
USE_MULTI_LL=false
USE_VISUALIZE=false

for hlactornum in 1 4
do
	for seed in 100 200 300 400 500 600 700 800 900 1000
	do

		for env in PointMaze_Medium_Diverse_GR FetchPick FetchPush FetchSlide HandManipulateEggRotate
		do


			for agent in Hierarchical #our methods
			do
				for trianglescale in  1.0 
				do

					while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
						sleep 10
					done
				

					gpu="${gpus[$idx]}"                 
					idx=$(((idx + 1) % gpu_n))          

					./run.sh $seed $gpu $agent $env $trianglescale $hlactornum $USE_CAPABLE $USE_MULTI_LL $USE_VISUALIZE> trainlogs/${hlactornum}_${env}_${agent}_${trianglescale}_${seed}_${dimlatent}_${USE_CAPABLE}_${USE_MULTI_LL}.log 2>&1 &
					
					sleep 10

				done
				

			done


		done

	done
done
wait
