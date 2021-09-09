#!/bin/bash
#SBATCH --mem=384g
#SBATCH -c1
#SBATCH --time=48:00:00
#SBATCH -J evalmbertengandakk
#SBATCH --output=/cs/labs/gabis/koren507/ancient-text-processing/slurms/%j.out
##SBATCH --killable
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=koren.lazar@mail.huji.ac.il

source /cs/labs/gabis/koren507/ancient-text-processing/venv/bin/activate
module load torch

K=30
SCRIPT_DIR=/cs/labs/gabis/koren507/ancient-text-processing/
MODEL_NAME="mbert_final_eng_and_akk"
MODEL_DIR="models/${MODEL_NAME}"

PREPROCESSED_TRAIN_AKK="data/preprocessed/new_train_eng.jsonl"
PREPROCESSED_TEST_AKK="data/preprocessed/new_test_eng.jsonl"
PREPROCESSED_TRAIN_ENG="data/preprocessed/new_train_eng.jsonl"
PREPROCESSED_TEST_ENG="data/preprocessed/new_test_eng.jsonl"

BATCH_SIZE=64
TOKENIZER_SUBDIR="akk_tokenizer"
METRICS_FILE="results/overall_metrics/metrics_${MODEL_NAME}.pickle"


python "${SCRIPT_DIR}akkadian_bert/main_bert.py" \
--do_eval \
--preprocessed_akk_train_json_file $PREPROCESSED_TRAIN_AKK \
--preprocessed_akk_test_json_file $PREPROCESSED_TEST_AKK \
--preprocessed_eng_train_json_file $PREPROCESSED_TRAIN_ENG \
--preprocessed_eng_test_json_file $PREPROCESSED_TEST_ENG \
--model_dir $MODEL_DIR \
--batch_size $BATCH_SIZE \
--hit_k $K \
--metrics_file $METRICS_FILE \
#--tokenizer_subdir $TOKENIZER_SUBDIR
