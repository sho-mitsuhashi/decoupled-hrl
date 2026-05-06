#!/bin/bash


#this experiment is compared againt Run_capable's baseline method

trap 'echo "Terminating process group..."; trap - SIGINT SIGTERM; kill -TERM -$$' SIGINT SIGTERM

mkdir trainlogs


gpus=(0)           # GPU list e.g.(0 1 2)
gpu_n=${#gpus[@]}        # GPU num
idx=0                    




MAX_JOBS=3 



USE_CAPABLE=false
USE_MULTI_LL=true
USE_VISUALIZE=false
hlactornum=2

for seed in 100 200 300 400 500 600 700 800 900 1000
do

	for env in PointMaze_Medium_Diverse_GR 
	do

		for agent in Hierarchical #our methods
		do
			for trianglescale in  1.0 0.0
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

wait
