#!/bin/bash
#SBATCH --job-name=lab2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=batch
#SBATCH --output=output_%j.out

module load intel/mpi4
cd /home/2024-01247/lab3

for size in 200 400 800 1200; do
    for proc in 1 2 4 8; do
        for rep in 1 2 3; do
            echo "Running: size=$size, proc=$proc, rep=$rep"
            mpirun -np $proc ./laba2_mpi $size $proc $rep
            sleep 1
        done
    done
done

python3 create_table.py
cat results_table.txt
