#!/bin/bash
#SBATCH --job-name=ollama_stepbystep
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=dino.wildi@ipw.uni-heidelberg.de

source /opt/bwhpc/common/etc/easybuild/enable_eb_modules
module load ollama/0.13.4-GCCcore-14.3.0-CUDA-12.9.1

export OLLAMA_HOST=0.0.0.0       # Serve on global interface
export OLLAMA_KEEP_ALIVE=2h      # Do not unload model (default is 5 minutes)
export OLLAMA_CONTEXT_LENGTH=16000

ollama serve
OLLAMA_PID=$!
echo "Ollama server started with PID $OLLAMA_PID"

# Wait for server to be ready
until curl -s http://localhost:11434 >/dev/null; do
  echo "Waiting for Ollama server to start..."
  sleep 2
done
echo "Ollama server is ready."

source $HOME/llm_coding/bin/activate
module load devel/python/3.13.3

python $HOME/projects/llm-coding/code/codebook_prompt.py

pkill ollama