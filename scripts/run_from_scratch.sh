#!/bin/bash
#SBATCH --mem=128g
#SBATCH -c1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH -J distakknolexicaltrain300epochs
#SBATCH --output=/cs/labs/gabis/koren507/ancient-text-processing/slurms/%j.out
##SBATCH --killable
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=koren.lazar@mail.huji.ac.il

source /cs/labs/gabis/koren507/ancient-text-processing/venv/bin/activate
module load torch

SCRIPT_DIR=/cs/labs/gabis/koren507/ancient-text-processing/
MODEL_DIR="models/distilbert_from_scratch_akk_300_epochs_no_lexical"

PREPROCESSED_TRAIN_AKK="data/preprocessed/new_train_akk_no_lexical.jsonl"
PREPROCESSED_TEST_AKK="data/preprocessed/new_test_akk.jsonl"
PREPROCESSED_TRAIN_ENG="data/preprocessed/new_train_eng.jsonl"
PREPROCESSED_TEST_ENG="data/preprocessed/new_test_eng.jsonl"

EPOCHS=300
BATCH_SIZE=4


python "${SCRIPT_DIR}akkadian_bert/main_bert.py" \
--from_scratch \
--epochs $EPOCHS \
--do_train \
--preprocessed_akk_train_json_file $PREPROCESSED_TRAIN_AKK \
--preprocessed_akk_test_json_file $PREPROCESSED_TEST_AKK \
--preprocessed_eng_train_json_file $PREPROCESSED_TRAIN_ENG \
--preprocessed_eng_test_json_file $PREPROCESSED_TEST_ENG \
--model_dir $MODEL_DIR \
--batch_size $BATCH_SIZE \
#--include_english
