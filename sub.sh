#!/bin/bash
#MSUB -S /bin/bash
#MSUB -o job.$PBS_JOBID
#MSUB -j oe
#MSUB -N Example
#MSUB -l nodes=1:ppn=2:gpus=1
#MSUB -l mem=2gb
#MSUB -l feature=epyc7713
#MSUB -q gpu
#MSUB -l walltime=20:00:00
#MSUB -d /home/tappay01/program


## change directory to working directory
cd $PBS_O_WORKDIR
echo "workdir: $PBS_O_WORKDIR"
echo "Hostname: ${HOSTNAME}"
nvidia-smi
## the work to be done, here we just report the nodes and cores that were allocated
echo "PBSNODEFILE: $PBS_NODEFILE"
NSLOTS=$(wc -l < $PBS_NODEFILE)
echo "running on $NSLOTS cores ..."


export PATH="/home/tappay01/anaconda3/bin/:$PATH"
eval "$(conda shell.bash hook)"

conda activate base
python main.py