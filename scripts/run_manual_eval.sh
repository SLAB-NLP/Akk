#!/bin/bash
#SBATCH --time=10-0
#SBATCH -c1
#SBATCH --mem=128g
#SBATCH -J akkman6eval
#SBATCH --output=/cs/labs/gabis/koren507/ancient-text-processing/slurms/manual_eval_akk_6_signs_29-04-2021-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=koren.lazar@mail.huji.ac.il

source /cs/labs/gabis/koren507/ancient-text-processing/venv/bin/activate
module load torch

MASKED_SEQUENCE_LENGTH=6
PARAMS_SUFFIX="with_hyphens_no_pseudowords_all_projects"
SCRIPT_DIR=/cs/labs/gabis/koren507/ancient-text-processing/
MODEL_DIR="models/mbert_final_eng_and_akk"
PREPROCESSED_TEST_AKK="data/preprocessed/new_test_akk.jsonl"
SAMPLED_PREPROCESSED_FILE="data/manual_evaluation/sampled_akk.jsonl"
MASKED_TEXTS_PATH="data/manual_evaluation/masked_akk_${MASKED_SEQUENCE_LENGTH}_signs.jsonl"
PREDICTIONS_FILE="data/manual_evaluation/predictions_akk_${MASKED_SEQUENCE_LENGTH}_signs.jsonl"
K_BEAMS=5
SAMPLE_SIZE=200
NUM_MASKS_IN_DOC=10

python "${SCRIPT_DIR}evaluation/manual_evaluations.py" \
--do_manual_eval_akkadian \
--preprocessed_akk_test_json_file $PREPROCESSED_TEST_AKK \
--sampled_preprocessed_akk_file $SAMPLED_PREPROCESSED_FILE \
--predictions_path $PREDICTIONS_FILE \
--masked_texts_path $MASKED_TEXTS_PATH \
--model_dir $MODEL_DIR \
--k_beams $K_BEAMS \
--masked_sequence_length $MASKED_SEQUENCE_LENGTH \
--sample_size $SAMPLE_SIZE \
--num_masks_in_doc $NUM_MASKS_IN_DOC \
# --predict_signs_only \
