#!/bin/bash
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:8
#SBATCH -J wiwitrain
# --mail-type=BEGIN,END,FAIL,TIME_LIMIT
# --mail-user=koren.lazar@mail.huji.ac.il

source /cs/labs/gabis/koren507/ancient-text-processing/venv/bin/activate
#source venv/bin/activate
module load torch

PARAMS_SUFFIX="with_hyphens_with_pseudowords_all_projects"
PREPROCESS_LOG_PATH="logs/preprocess_log_${PARAMS_SUFFIX}.log"
PREPROCESSED_TRAIN_AKK="data/train_jsons_akk_preprocessed_${PARAMS_SUFFIX}.jsonl"
PREPROCESSED_TEST_AKK="data/test_jsons_akk_preprocessed_${PARAMS_SUFFIX}.jsonl"
PREPROCESSED_TRAIN_ENG="data/train_jsons_eng_preprocessed_${PARAMS_SUFFIX}.jsonl"
PREPROCESSED_TEST_ENG="data/test_jsons_eng_preprocessed_${PARAMS_SUFFIX}.jsonl"


SCRIPT_DIR=/cs/labs/gabis/koren507/ancient-text-processing/
MODEL_DIR="models/mbert_${PARAMS_SUFFIX}"
#MODEL_NAME=tiny-bert-10000-epochs.pkl
#PREPROCESSED_TEST=data/tiny_test.jsonl
BERT_PLOT_PATH="plots/${PARAMS_SUFFIX}.jpg"
BERT_LOG_PATH="logs/bert_log_${PARAMS_SUFFIX}.log"
EPOCHS=20
BATCH_SIZE=4
PROJECT_NAMES="all"


#python "${SCRIPT_DIR}preprocessing/main_preprocess.py" --log_file $PREPROCESS_LOG_PATH --train_akk_preprocessed_data_file $PREPROCESSED_TRAIN_AKK --test_akk_preprocessed_data_file $PREPROCESSED_TEST_AKK --train_eng_preprocessed_data_file $PREPROCESSED_TRAIN_ENG --test_eng_preprocessed_data_file $PREPROCESSED_TEST_ENG --project_names $PROJECT_NAMES --use_pseudo_words


python "${SCRIPT_DIR}akkadian_bert/train_bert.py" --log_file $BERT_LOG_PATH --plot_file $BERT_PLOT_PATH --epochs $EPOCHS --do_train --preprocessed_akk_train_json_file $PREPROCESSED_TRAIN_AKK --preprocessed_akk_test_json_file $PREPROCESSED_TEST_AKK --preprocessed_eng_train_json_file $PREPROCESSED_TRAIN_ENG --preprocessed_eng_test_json_file $PREPROCESSED_TEST_ENG --mbert --model_dir $MODEL_DIR --batch_size $BATCH_SIZE
#python "${SCRIPT_DIR}akkadian_bert/train_bert.py"
