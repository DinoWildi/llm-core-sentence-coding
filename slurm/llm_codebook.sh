#!/bin/bash
#SBATCH --job-name=ollama_codebook
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_a100_short
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=dino.wildi@ipw.uni-heidelberg.de

# GPU Monitoring setup
GPU_LOG="gpu_logs/gpu_usage_log${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 5 >> ${GPU_LOG} &
NVIDIA_SMI_PID=$!
echo "GPU Monitoring started with PID ${NVIDIA_SMI_PID}, logging to ${GPU_LOG}"

# Ollama setup
source /opt/bwhpc/common/etc/easybuild/enable_eb_modules
module load ollama/0.32.5-GCCcore-14.3.0-CUDA-12.9.1

export OLLAMA_HOST=0.0.0.0       # Serve on global interface
export OLLAMA_KEEP_ALIVE=2h      # Do not unload model (default is 5 minutes)
export OLLAMA_CONTEXT_LENGTH=10000

ollama serve >> ollama_logs/ollama_server_log${SLURM_JOB_ID}.log 2>&1  &
OLLAMA_PID=$!
echo "Ollama server started with PID $OLLAMA_PID"

sleep 10

# Wait for server to be ready
until curl -s http://localhost:11434 >/dev/null; do
  echo "Waiting for Ollama server to start..."
  sleep 2
done
echo "Ollama server is ready."

source $HOME/llm_coding/bin/activate
module load devel/python/3.13.3

python $HOME/projects/llm-coding/code/codebookprompt_hpc.py

pkill ollama
kill ${NVIDIA_SMI_PID} 2>/dev/null
echo "GPU usage log saved to ${GPU_LOG}"
