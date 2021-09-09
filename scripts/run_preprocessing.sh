#!/bin/bash
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=48:00:00
#SBATCH -J preprocessing
#SBATCH --output=/cs/labs/gabis/koren507/ancient-text-processing/slurms/%j.out

source /cs/labs/gabis/koren507/ancient-text-processing/venv/bin/activate

PARAMS_SUFFIX="all_projects"

SCRIPT_DIR=/cs/labs/gabis/koren507/ancient-text-processing/
MODEL_DIR="models/mbert_${PARAMS_SUFFIX}"

PREPROCESS_LOG_PATH="${MODEL_DIR}/preprocess_log_${PARAMS_SUFFIX}.log"
PREPROCESSED_TRAIN_AKK="${MODEL_DIR}/train_jsons_akk_preprocessed_${PARAMS_SUFFIX}.jsonl"
PREPROCESSED_TEST_AKK="${MODEL_DIR}/test_jsons_akk_preprocessed_${PARAMS_SUFFIX}.jsonl"
PREPROCESSED_TRAIN_ENG="${MODEL_DIR}/train_jsons_eng_preprocessed_${PARAMS_SUFFIX}.jsonl"
PREPROCESSED_TEST_ENG="${MODEL_DIR}/test_jsons_eng_preprocessed_${PARAMS_SUFFIX}.jsonl"
PROJECT_NAMES="all"


python "${SCRIPT_DIR}preprocessing/main_preprocess.py" \
--do_scraping \
--do_preprocessing \
--log_file $PREPROCESS_LOG_PATH \
--train_akk_preprocessed_data_file $PREPROCESSED_TRAIN_AKK \
--test_akk_preprocessed_data_file $PREPROCESSED_TEST_AKK \
--train_eng_preprocessed_data_file $PREPROCESSED_TRAIN_ENG \
--test_eng_preprocessed_data_file $PREPROCESSED_TEST_ENG \
--project_names $PROJECT_NAMES \
--remove_subscripts \
--remove_superscripts \
