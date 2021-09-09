#!/bin/bash
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=72:00:00
#SBATCH -J chaueval
#SBATCH --output=/cs/labs/gabis/koren507/ancient-text-processing/slurms/chau_eval_01-04-2021-%j.out

source /cs/labs/gabis/koren507/ancient-text-processing/venv/bin/activate
module load torch

SCRIPT_DIR=/cs/labs/gabis/koren507/ancient-text-processing/

python "${SCRIPT_DIR}evaluation/chau_evaluation.py" \
--vocab /cs/usr/koren507/gabis_lab/ancient-text-processing/models/mt/vocab-mbert-5000.txt \
--model /cs/usr/koren507/gabis_lab/ancient-text-processing/models/mt/mt-va-15 \
--data /cs/usr/koren507/gabis_lab/ancient-text-processing/models/mt/test_tiny.txt
