#!/bin/bash
#SBATCH --workdir=/home/tacucumides/storage
#SBATCH --ntasks=1
#SBATCH --job-name=nbfnet-train
#SBATCH --nodelist=hydra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tacucumides@uc.cl
#SBATCH --output=/home/tacucumides/storage/NBFNet/logs
#SBATCH --gres=gpu:TitanRTX:1
#SBATCH --cpus=2
#SBATCH --partition=ialab-high

pwd; hostname; date
echo "Start"
echo $(pwd)
cd /home/tacucumides/storage
source miniconda3/etc/profile.d/conda.sh
conda activate gnn-qe
cd /home/tacucumides/storage/NBFNet
python script/run.py -c config/knowledge_graph/fb15k237.yaml --gpus [0] --version v1
