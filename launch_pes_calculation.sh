#!/bin/bash

for idx in {0..99}; do
	cp launch_pwscf.sh 3Cu_$idx/launch_pwscf.sh
	cd 3Cu_$idx/
	qsub ./launch_pwscf.sh
	cd ../
done 
