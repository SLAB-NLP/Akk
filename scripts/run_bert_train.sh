#!/bin/bash
#SBATCH --mem=128g
#SBATCH -c4
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:4
# --mail-type=BEGIN,END,FAIL,TIME_LIMIT
# --mail-user=koren.lazar@mail.huji.ac.il

source /cs/labs/gabis/koren507/ancient-text-processing/venv/bin/activate
module load torch

SCRIPT_DIR=/cs/labs/gabis/koren507/ancient-text-processing/
MODEL_DIR=/cs/labs/gabis/koren507/ancient-text-processing/models/tiny_bert
MODEL_NAME=tiny-bert-10000-epochs.pkl
PREPROCESSED_TEST=data/tiny_test.jsonl
PLOT_PATH=plots/first_plot.jpg
LOG_PATH=logs/first_log.log
EPOCHS=100

python "${SCRIPT_DIR}akkadian_bert/train_bert.py" --log_file $LOG_PATH --plot_file $PLOT_PATH --epochs $EPOCHS --do_train --do_eval

#--model_name $MODEL_NAME --pkl_dir $MODEL_DIR --data_path $TRAIN_PATH --train --num_iters 100000
#echo Running test:
#python "${SCRIPT_DIR}linear_classifier.py" --model_name $MODEL_NAME --pkl_dir $MODEL_DIR --data_path $TEST_PATH --result_path #$OUTPUT --test --confusion_matrix $PLOT_PATH
