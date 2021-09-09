#!/bin/bash
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=48:00:00
#SBATCH -J winoscr
#SBATCH --output=/cs/labs/gabis/koren507/ancient-text-processing/slurms/scraping_wino_27-04-2021-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=koren.lazar@mail.huji.ac.il

source /cs/labs/gabis/koren507/ancient-text-processing/venv/bin/activate
module load torch

PARAMS_SUFFIX="with_hyphens_no_pseudowords_all_projects"
SCRIPT_DIR=/cs/labs/gabis/koren507/ancient-text-processing/
MODEL_DIR="models/mbert_final"

PREPROCESS_LOG_PATH="${MODEL_DIR}/preprocess_log_${PARAMS_SUFFIX}.log"
PROJECT_NAMES="all"


python "${SCRIPT_DIR}preprocessing/main_preprocess.py" \
--log_file $PREPROCESS_LOG_PATH \
--do_scraping \
--raw_akk_data_file data/new_akk_data1.jsonl \
--raw_eng_data_file data/new_eng_data.jsonl \
--project_names $PROJECT_NAMES \
