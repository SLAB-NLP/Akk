#!/bin/bash
#SBATCH --mem=64g
#SBATCH -c4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:8
#SBATCH -J mbertalltrain
#SBATCH --output=/cs/labs/gabis/koren507/ancient-text-processing/slurms/%j.out
##SBATCH --killable
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=koren.lazar@mail.huji.ac.il

source /cs/labs/gabis/koren507/ancient-text-processing/venv/bin/activate
module load torch

PARAMS_SUFFIX="all_projects"
SCRIPT_DIR=/cs/labs/gabis/koren507/ancient-text-processing/
MODEL_DIR="models/mbert_${PARAMS_SUFFIX}"

PREPROCESSED_TRAIN_AKK="${MODEL_DIR}/train_jsons_akk_preprocessed_${PARAMS_SUFFIX}.jsonl"
PREPROCESSED_TEST_AKK="${MODEL_DIR}/test_jsons_akk_preprocessed_${PARAMS_SUFFIX}.jsonl"
PREPROCESSED_TRAIN_ENG="${MODEL_DIR}/train_jsons_eng_preprocessed_${PARAMS_SUFFIX}.jsonl"
PREPROCESSED_TEST_ENG="${MODEL_DIR}/test_jsons_eng_preprocessed_${PARAMS_SUFFIX}.jsonl"

BERT_PLOT_PATH="${MODEL_DIR}/train_plot.jpg"
BERT_LOG_PATH="${MODEL_DIR}/train_log.log"
EPOCHS=20
BATCH_SIZE=2


python "${SCRIPT_DIR}akkadian_bert/main_bert.py" \
--log_file $BERT_LOG_PATH \
--plot_file $BERT_PLOT_PATH \
--epochs $EPOCHS \
--do_train \
--preprocessed_akk_train_json_file $PREPROCESSED_TRAIN_AKK \
--preprocessed_akk_test_json_file $PREPROCESSED_TEST_AKK \
--preprocessed_eng_train_json_file $PREPROCESSED_TRAIN_ENG \
--preprocessed_eng_test_json_file $PREPROCESSED_TEST_ENG \
--model_dir $MODEL_DIR \
--batch_size $BATCH_SIZE \
--include_english
