#!/bin/bash
#SBATCH --workdir=/home/tacucumides/storage
#SBATCH --ntasks=1
#SBATCH --job-name=nbfnet-test
#SBATCH --nodelist=scylla
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tacucumides@uc.cl
#SBATCH --output=/home/tacucumides/storage/NBFNet/logs/%A.log
#SBATCH --gres=gpu:1
#SBATCH --cpus=2
#SBATCH --partition=ialab-high

pwd; hostname; date
echo "Start"
echo $(pwd)
cd /home/tacucumides/storage
source miniconda3/etc/profile.d/conda.sh
conda activate nbfnet
cd /home/tacucumides/storage/NBFNet
python script/run.py -c config/knowledge_graph/fb15k237-test.yaml --checkpoint ../2024-01-14-12-42-34/model_epoch_14.pth  --gpus [0] --version v1
