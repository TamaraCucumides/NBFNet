#!/bin/bash
#SBATCH --workdir=/home/tacucumides/storage
#SBATCH --ntasks=1
#SBATCH --job-name=nbfnet-save
#SBATCH --nodelist=scylla
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tacucumides@uc.cl
#SBATCH --output=/home/tacucumides/storage/NBFNet/logs/%A.log
#SBATCH --gres=gpu:0
#SBATCH --cpus=2
#SBATCH --partition=ialab-high

pwd; hostname; date
echo "Start"
echo $(pwd)
cd /home/tacucumides/storage
source miniconda3/etc/profile.d/conda.sh
conda activate nbfnet
cd /home/tacucumides/storage/NBFNet
python script/save_predictions.py -c config/knowledge_graph/fb15k237_visualize.yaml  --checkpoint ../pesos/model_epoch_14.pth  --gpus null --version v1




