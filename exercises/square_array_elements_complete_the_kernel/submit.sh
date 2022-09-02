#!/bin/bash

#SBATCH -A TRN001
#SBATCH -J complete_the_kernel
#SBATCH -N 1
#SBATCH -t 10
#SBATCH -p batch
#SBATCH -o %x-%j.out

srun -n1 -c1 --gpus=1 ./square_array_elements

