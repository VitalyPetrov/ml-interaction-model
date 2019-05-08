#!/bin/sh
#PBS -l nodes=1:ppn=16,walltime=3:00:00
#PBS -N pes_espresso
#PBS -q batch
#PBS -j oe
#PBS -V 

NUM_MPI_THREADS=16
ESPRESSO_BIN="/opt/qe/6.2.1/mpi/intel/2017/GCC/5.3.1/bin/pw.x"

cd ${PBS_O_WORKDIR}
module load MPI/impi/2017.0.4 MKL/2017.4.239 Compiler/GCC/5.3.1

mpirun -np $NUM_MPI_THREADS $ESPRESSO_BIN -in espresso.pwi -o espresso.pwo
