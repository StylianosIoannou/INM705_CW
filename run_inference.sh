#!/bin/bash
#SBATCH --job-name inm705                      # Job name
#SBATCH --partition=gengpu                         # Select the correct partition.
#SBATCH --nodes=1                                # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=4                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --mem=24GB                                 # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=01:00:00                            # Expected ammount of time to run Time limit hrs:min:sec
#SBATCH --gres=gpu:1                               # Use one gpu.
#SBATCH -e inference_error_%j.txt                  # Error log
#SBATCH -o inference_output_%j.txt                 # Output log

#Enable modules command

source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu


export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export WANDB_API_KEY=43d492ebc66777f5a74d183cc0be328e30fb528a

echo $WANDB_API_KEY

# Activate your virtual environment
#source venv/bin/activate


python --version
#module load libs/nvidia-cuda/11.2.0/bin

#wandb login $WANDB_API_KEY --relogin
#pip freeze
#Run your script.
python3 Inference.py
