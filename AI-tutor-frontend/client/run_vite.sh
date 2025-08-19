#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1
#SBATCH --ntasks=1
#SBATCH --time=0-100:00
#SBATCH --mem=80G
#SBATCH --output=%N-%j.out
#SBATCH --qos=batch-long
#SBATCH --output=logs/%N-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=yashy.yash@deakin.edu.au

npm run dev
 
