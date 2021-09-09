import argparse
from typing import List
import csv

from transformers import (
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
)

from pathlib import Path
import logging
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from akkadian_bert.write_bert import write_bert_files
from preprocessing.main_preprocess import MISSING_SIGN_CHAR
from akkadian_bert.data_collators_bert import DataCollatorForLanguageModelingAkkadian
from akkadian_bert.datasets_bert import ORACCDataset
from akkadian_bert.tokenize_bert import train_tokenizer, inject_new_tokens_to_tokenizer
from akkadian_bert.evaluate_bert import compute_mrr_and_all_hit_ks

SPECIAL_TOKENS_NUM = 5

VOCAB_SIZE = 2_000

ROBERTA_UNUSED_TOKENS_NUM = 99

MAIN_TOKENIZER_DIRNAME = "main_tokenizer"
AKK_TOKENIZER_DIRNAME = "akk_tokenizer"
AKKADIAN_BERT_DIRNAME = "akkadian_bert"
BERT_TOTAL_TRAIN_INPUT_TXT_FILE = "bert_total_train_dataset.txt"
BERT_TRAIN_INPUT_TXT_FILE = "bert_train_dataset.txt"
BERT_VAL_INPUT_TXT_FILE = "bert_val_dataset.txt"
BERT_TEST_INPUT_TXT_FILE = "bert_test_dataset.txt"
LOG_FILE = "logs/train_bert.log"

BERT_AKK_TRAIN_FILE = "bert_akk_train_dataset.txt"
BERT_ENG_TRAIN_FILE = "bert_eng_train_dataset.txt"
BERT_AKK_TEST_FILE = "bert_akk_test_dataset.txt"
BERT_ENG_TEST_FILE = "bert_eng_test_dataset.txt"


def get_bert_path(model_dir, filename):
    return f"./{model_dir}/{filename}"


logger = None


def bert_from_scratch(preprocessed_akk_train_file: str, preprocessed_eng_train_file: str,
                      preprocessed_akk_test_file: str, model_dir: str, epochs: int, batch_size: int) -> None:
    """
    This function trains from scratch a BERT-based model.
    First, it trains a tokenizer and save it, and then use it to a train a BERT-based model.

    :param preprocessed_akk_train_file: A path to a given preprocessed jsonl Akkadian train data file
    :param preprocessed_akk_test_file: A path to a given preprocessed jsonl Akkadian test data file
    :param preprocessed_eng_train_file: A path to a given preprocessed jsonl English train data file
    :param epochs: The number of epochs to train the model.
    :param model_dir: The model's directory name
    :param batch_size: The size of each training batch.
    """
    bert_akk_train_path = f"./{model_dir}/{BERT_AKK_TRAIN_FILE}"
    bert_train_path = f"./{model_dir}/{BERT_TRAIN_INPUT_TXT_FILE}"
    bert_test_path = f"./{model_dir}/{BERT_TEST_INPUT_TXT_FILE}"

    write_bert_files(
        preprocessed_akk_file=preprocessed_akk_train_file,
        preprocessed_eng_file=preprocessed_eng_train_file,
        bert_akk_file=bert_akk_train_path,
        bert_eng_file=f"./{model_dir}/{BERT_ENG_TRAIN_FILE}",
        bert_final_path=bert_train_path,
    )
    write_bert_files(
        preprocessed_akk_file=preprocessed_akk_test_file,
        preprocessed_eng_file=None,
        bert_akk_file=f"./{model_dir}/{BERT_AKK_TEST_FILE}",
        bert_eng_file=f"./{model_dir}/{BERT_ENG_TEST_FILE}",
        bert_final_path=bert_test_path,
    )

    train_tokenizer(model_dir, VOCAB_SIZE, bert_akk_train_path)

    config = DistilBertConfig(
        vocab_size=VOCAB_SIZE,
        n_heads=2,
        n_layers=2,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        dim=128,
        hidden_dim=4 * 128,
    )

    tokenizer = BertTokenizerFast.from_pretrained(f"./{model_dir}/akk_tokenizer", max_len=512)

    model = AutoModelForMaskedLM.from_config(config)
    logging.info(f"The model's number of parameters is {model.num_parameters()}")

    train_model_mlm(epochs, model, model_dir, tokenizer, bert_train_path, bert_test_path, batch_size)


def train_model_mlm(epochs: int, model: AutoModelForMaskedLM, model_dir: str, tokenizer: BertTokenizerFast,
                    bert_train_path: str, bert_test_path: str, batch_size: int) -> None:
    """
    This function trains a BERT-based model on the masked language modelling task.

    :param epochs: The number of epochs to train
    :param model: A given model
    :param model_dir: The directory of the given model
    :param tokenizer: A given tokenizer
    :param bert_train_path: A path to the training data file
    :param bert_test_path: A path to the test data file
    :param batch_size: The size of eah training batch
    """
    train_dataset = ORACCDataset(
        file_path=bert_train_path,
        tokenizer=tokenizer,
        block_size=128,  # TODO: can and should we split documents?
        missing_sign_encoding=tokenizer.get_vocab()[MISSING_SIGN_CHAR],
        encode_only_first_token_in_word=False,
        ignore_missing=True,
    )

    test_dataset = ORACCDataset(
        file_path=bert_test_path,
        tokenizer=tokenizer,
        block_size=128,
        missing_sign_encoding=tokenizer.get_vocab()[MISSING_SIGN_CHAR],
        encode_only_first_token_in_word=False,
        ignore_missing=True,
    )

    data_collator = DataCollatorForLanguageModelingAkkadian(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=f"./{model_dir}",
        do_eval=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=model_dir,
        gradient_accumulation_steps=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_mrr_and_all_hit_ks,
    )
    logging.info("Started training!")
    trainer.train()

    # logging.info("Started evaluation!")
    # metrics = trainer.evaluate()
    # logging.info(metrics)
    # print(metrics)

    trainer.save_model(f"./{model_dir}")


def m_bert(preprocessed_akk_train_file, preprocessed_eng_train_file, preprocessed_akk_test_file, model_dir, epochs,
           batch_size):
    """
    This function prepares a Multilingual BERT model for further pre-training in train_model_mlm function.

    :param preprocessed_akk_train_file:
    :param preprocessed_eng_train_file:
    :param preprocessed_akk_test_file:
    :param model_dir:
    :param epochs:
    :param batch_size:
    :return:
    """
    bert_akk_train_path = f"./{model_dir}/{BERT_AKK_TRAIN_FILE}"
    bert_train_path = f"./{model_dir}/{BERT_TRAIN_INPUT_TXT_FILE}"
    bert_test_path = f"./{model_dir}/{BERT_TEST_INPUT_TXT_FILE}"

    write_bert_files(
        preprocessed_akk_file=preprocessed_akk_train_file,
        preprocessed_eng_file=preprocessed_eng_train_file,
        bert_akk_file=bert_akk_train_path,
        bert_eng_file=f"./{model_dir}/{BERT_ENG_TRAIN_FILE}",
        bert_final_path=bert_train_path,
    )
    write_bert_files(
        preprocessed_akk_file=preprocessed_akk_test_file,
        preprocessed_eng_file=None,
        bert_akk_file=f"./{model_dir}/{BERT_AKK_TEST_FILE}",
        bert_eng_file=f"./{model_dir}/{BERT_ENG_TEST_FILE}",
        bert_final_path=bert_test_path,
    )

    train_tokenizer(model_dir, ROBERTA_UNUSED_TOKENS_NUM * 10, bert_akk_train_path)
    # akk_tokenizer = AutoTokenizer.from_pretrained(model_dir, vocab=f"./{model_dir}/akk_tokenizer")
    with open(f"./{model_dir}/akk_tokenizer/vocab.txt", "r", encoding='utf-8') as f_vocab:
        akk_tokens = f_vocab.read().splitlines()

    # Load "bert-base-multilingual-cased" model.
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Augment to its vocab.json #ROBERTA_UNUSED_TOKENS_NUM most frequent tokens in Akkadian.
    # with open(f"./{model_dir}/vocab.json", "r", encoding='utf-8') as f_akk_vocab:
    #     akk_tokens = list(json.load(f_akk_vocab).keys())
    # TODO: can we extend an existing tokenizer?
    inject_new_tokens_to_tokenizer(akk_tokens, tokenizer, tokenizer.all_special_tokens)
    tokenizer.save_pretrained(f"./{model_dir}/{MAIN_TOKENIZER_DIRNAME}/")
    tokenizer_fast = BertTokenizerFast.from_pretrained(f"./{model_dir}/{MAIN_TOKENIZER_DIRNAME}/")
    # TODO: is tokenizer_fast needed?

    # bert_train_divided_path = f"./{model_dir}/temp.txt"
    # with open(bert_train_divided_path, 'w') as f_out, open(bert_train_path, 'r') as f_in:
    #     for line in f_in:
    #         for short_line in split_input_by_block_size(line, tokenizer_fast, 128):
    #             f_out.write(short_line)

    # Train the model with the input texts.
    model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
    train_model_mlm(
        epochs=epochs,
        model=model,
        model_dir=model_dir,
        tokenizer=tokenizer_fast,
        bert_train_path=bert_train_path,
        bert_test_path=bert_test_path,
        batch_size=batch_size,
    )


def split_input_by_block_size(input: str, tokenizer: BertTokenizerFast, block_size: int) -> List[str]:
    # TODO: Validate correctness, check for splitting by last sentence
    new_inputs = list()
    encodings = tokenizer.encode(input)
    while len(encodings) > block_size:
        new_inputs.append(tokenizer.decode(encodings[:block_size]))
        encodings = encodings[block_size:]
    new_inputs.append(tokenizer.decode(encodings))
    return new_inputs


def tokens_stats(input_file, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(f"./{model_dir}", max_len=512)

    token_count = 0
    wordpiece_count = 0
    with open(input_file, "r", encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            orig_tokens = line.strip().split()
            for token in orig_tokens:
                token_count += 1
                wordpieces = tokenizer.tokenize(token)
                wordpiece_count += len(wordpieces)

            if i % 100 == 0:
                print(f"{input_file}: {token_count} tokens, {wordpiece_count} wordpieces, ratios is then "
                      f"{wordpiece_count / token_count}")
