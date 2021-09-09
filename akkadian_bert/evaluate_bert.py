import argparse
import glob
import json
import logging
import os
import pickle
import random
import re
from copy import deepcopy
from enum import Enum
from typing import List
from operator import itemgetter
import string
import torch

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import (
    Pipeline,
    PreTrainedTokenizer,
    pipeline,
    BertForMaskedLM,
    BertTokenizer,
    AutoModelForMaskedLM,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)

from akkadian_bert.write_bert import write_bert_files
from akkadian_bert.data_collators_bert import DataCollatorForLanguageModelingAkkadian
from akkadian_bert.datasets_bert import ORACCDataset
from preprocessing.main_preprocess import MISSING_SIGN_CHAR, remove_squared_brackets, SUPERSCRIPTS_TO_UNICODE_CHARS, \
    INTENTIONAL_HOLE_IN_CUNEIFORM, _remove_redundant_parts, add_properties_to_data, Certainty
from akkadian_bert.utils import calc_wind_around_ind
from preprocessing.scraping import JSONS_DIR, _load_json_from_path, get_raw_akk_text_from_json, \
    get_raw_text_akk_from_html

MBERT_BASE_FAKE_MODEL_DIR = 'models/mbert_base'

MISSING_SIGN_IN_ANNOTATION = "\N{cross mark}"

np.set_printoptions(precision=5, suppress=True)

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
BERT_TINY_DATASET = "models/mbert_with_hyphens_no_pseudowords_all_projects/bert_tiny_test_dataset.txt"
BERT_TEST_INPUT_TXT_FILE = "bert_test_dataset.txt"
BERT_AKK_TEST_FILE = "bert_akk_test_dataset.txt"
hit_k = 30


class ORACCLanguage(Enum):
    Akkadian = 0
    English = 1


def update_hit_k(k):
    global hit_k
    hit_k = k


def generate_evaluations(input_txt_file, model_dir, vocab_file, k=10):
    fill_mask = pipeline(
        "fill-mask",
        model=BertForMaskedLM.from_pretrained(model_dir),
        tokenizer=BertTokenizer(vocab_file=vocab_file, do_lower_case=False, do_basic_tokenize=True),
        top_k=100
    )
    mrr = hit_k = 0
    tokens_counter = 0
    with open(input_txt_file, "r", encoding='utf-8') as f_in:

        for line in tqdm(f_in):
            words_in_line = line.split()
            words_in_line_original = deepcopy(words_in_line)
            for i in range(len(words_in_line)):
                # Mask the ith word:
                words_in_line[i] = fill_mask.tokenizer.mask_token

                # Calculate the words that the model will see when predicting the masked word:
                min_index, max_index = calc_wind_around_ind(i, len(words_in_line))

                # Get the candidate words to fill in the mask word:
                cur_pred = fill_mask(" ".join(words_in_line[min_index:max_index]))
                cur_tokens_str = np.array([candidate['token_str'] for candidate in cur_pred])

                # Update the metrics
                condition = np.logical_or(cur_tokens_str == words_in_line_original[i],
                                          cur_tokens_str == ("Ġ" + words_in_line_original[i]))
                indices = np.argwhere(condition)
                if len(indices):
                    mrr += 1 / (indices[0][0] + 1)
                    if indices[0][0] < k:
                        hit_k += 1
                words_in_line[i] = words_in_line_original[i]
                tokens_counter += 1
            logging.info(f"The MRR is {mrr / tokens_counter}")
            logging.info(f"The hit@{k} is {hit_k / tokens_counter}")
    if tokens_counter != 0:
        mrr /= tokens_counter
        hit_k /= tokens_counter
    logging.info(f"The mrr is {mrr}")
    logging.info(f"The hit@{k} is {hit_k}")


def generate_evaluations_by_certainty(preprocessed_input_file: str, plot_path, model_dir: str, k: int = 10):
    """
    This function generates evaluations of MRR and hit@k for a given preprocessed input jsonl file.
    It assumes that the preprocessed input file has a certain format: evey line is a json that has a
    "preprocessed_words" field which is a list containing the text of a document (in tuples).

    :param preprocessed_input_file: A given preprocessed input jsonl file
    :param model_dir: The model's directory name
    :param plot_path: A path to save the generated plot
    :param k: The k parameter for hit@k metric
    """
    model = AutoModelForMaskedLM.from_pretrained(f"./{model_dir}")
    # tokenizer = AutoTokenizer.from_pretrained(f"./{model_dir}")
    tokenizer_fast = BertTokenizerFast.from_pretrained(f"./{model_dir}")

    # fill_mask = pipeline(
    #     "fill-mask",
    #     model=f"./{model_dir}",
    #     tokenizer=tokenizer_fast,  # if tokenizer_fast else f"./{model_dir}",
    #     topk=100,
    # )
    # - 1 as we don't evaluate missing words
    certainties_levels_num = len(Certainty) - 1
    mrrs, hit_ks = [[] for _ in range(certainties_levels_num)], np.zeros(certainties_levels_num)
    with open(preprocessed_input_file, "r", encoding='utf-8') as f_in:
        for line in tqdm(f_in):
            preprocessed_words_triplets = json.loads(line)["preprocessed_words"]
            words_in_line = [word_triplet["preprocessed"] for word_triplet in preprocessed_words_triplets]
            words_in_line_original = deepcopy(words_in_line)

            # Get the candidate words to fill in the mask word:
            # input_txt = words_in_line[min_index:max_index]

            inputs = tokenizer_fast(
                words_in_line,
                truncation=True,  # TODO: should we truncate?
                padding=True,
                is_split_into_words=True,
                return_tensors='pt')
            encodings = inputs.data['input_ids']
            original_encodings = deepcopy(encodings)

            arr_offsets = np.array(inputs.encodings[0].offsets)
            first_tokens_indices = np.nonzero((arr_offsets[:, 0] == 0) & (arr_offsets[:, 1] != 0))[0]

            for i in range(min(len(words_in_line), 1000)):
                # The certainty of the current masked word
                certainty = Certainty[preprocessed_words_triplets[i]["certainty"]]
                if certainty == Certainty.MISSING:
                    continue

                # Mask the ith word:
                words_in_line[i] = tokenizer_fast.mask_token  # * num_tokens_to_mask

                # Calculate the words that the model will see when predicting the masked word:
                # min_index, max_index = calc_wind_around_ind(i, len(words_in_line))
                # TODO: truncation from what side?

                encodings[0, first_tokens_indices[i]] = tokenizer_fast.mask_token_id
                # enc_labels = get_enc_labels(inputs, tokenizer.get_vocab()[MISSING_SIGN_CHAR], encode_only_first_token_in_word=True)

                outputs = model(encodings)
                # cur_pred = fill_mask(" ".join(words_in_line[min_index:max_index]))

                predictions = outputs[0]
                predicted_tokens = outputs[0][0].sort(dim=-1, descending=True)[1][i + 1, :][:100]
                # predicted_tokens = list()
                # temp = np.array([candidate['token'] for candidate in cur_pred])

                # sorted_preds, sorted_idx = predictions[0].sort(dim=-1, descending=True)
                # for k in range(100):
                #     predicted_index = [sorted_idx[i, k].item() for i in range(100)]
                #     predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(1, 100)]
                #     predicted_tokens.append(predicted_token)

                # cur_tokens_str = np.array([candidate['token_str'] for candidate in cur_pred
                #                            if candidate['token_str'] != MISSING_SIGN_CHAR])

                # Update the metrics
                condition = predicted_tokens == original_encodings[0, i + 1]
                indices = torch.nonzero(condition)
                if len(indices):
                    mrrs[certainty.value].append(1 / (torch.IntTensor.item(indices[0]) + 1))
                    if indices[0][0] < k:
                        hit_ks[certainty.value] += 1
                else:
                    mrrs[certainty.value].append(0)
                words_in_line[i] = words_in_line_original[i]
                encodings[0, i + 1] = original_encodings[0, i + 1]
            hit_ks, mrrs_avgs = np.divide((hit_ks, [sum(mrr) for mrr in mrrs]), [max(len(mrr), 1) for mrr in mrrs])
            logging.info(f"The MRR is {mrrs_avgs}")
            logging.info(f"The hit@{k} is {hit_ks}")
            # mrrs_avgs = np.divide([sum(mrr) for mrr in mrrs], [len(mrr) for mrr in mrrs])
            mrrs_stds = [np.std(mrr) if len(mrr) else 0 for mrr in mrrs]
            plot_metrics(hit_ks, k, mrrs_avgs, mrrs_stds, plot_path)

            fig = plt.figure()
            plt.bar([certainty.name for certainty in Certainty if certainty != Certainty.MISSING],
                    [len(mrr) for mrr in mrrs])
            fig.autofmt_xdate()
            plt.savefig(f"./{model_dir}/certainty_hist.jpg")
            plt.close()


def eval_compute_metrics(preprocessed_akk_input_file: str, model_dir: str, batch_size, shorten,
                         tokenizer_subdir='main_tokenizer', debug: bool = False):
    """
    This function evaluated the MRR and hit@k given a trained model and a preprocessed input file.

    :param shorten:
    :param tokenizer_subdir:
    :param preprocessed_akk_input_file:
    :param model_dir:
    :param batch_size:
    :param debug:
    :return:
    """
    # TODO: is tokenizer_fast needed?
    if model_dir == MBERT_BASE_FAKE_MODEL_DIR:
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        missing_sign_char = 'x'

    else:
        model = AutoModelForMaskedLM.from_pretrained(f"./{model_dir}")
        tokenizer = BertTokenizerFast.from_pretrained(f"./{model_dir}/{tokenizer_subdir}")
        missing_sign_char = MISSING_SIGN_CHAR

    logging.info(f"The missing character is {missing_sign_char}")

    if debug:
        bert_test_path = BERT_TINY_DATASET
    else:
        bert_test_path = f"./{model_dir}/{BERT_TEST_INPUT_TXT_FILE}"
        write_bert_files(
            preprocessed_akk_file=preprocessed_akk_input_file,
            preprocessed_eng_file=None,
            bert_akk_file=f"./{model_dir}/{BERT_AKK_TEST_FILE}",
            bert_eng_file=None,
            bert_final_path=bert_test_path,
            shorten=shorten,
        )

    test_dataset = ORACCDataset(
        file_path=bert_test_path,
        tokenizer=tokenizer,
        block_size=128,
        missing_sign_encoding=tokenizer.get_vocab()[missing_sign_char],
        encode_only_first_token_in_word=False,
        ignore_missing=True,
    )

    data_collator = DataCollatorForLanguageModelingAkkadian(tokenizer=tokenizer, missing_sign=missing_sign_char)

    training_args = TrainingArguments(
        output_dir=f"./{model_dir}",
        do_eval=True,
        per_device_eval_batch_size=batch_size,
        overwrite_output_dir=True,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=model_dir,
        no_cuda=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_dataset,
        compute_metrics=compute_mrr_and_all_hit_ks,
    )
    metrics = trainer.evaluate()
    logging.info(metrics)
    return metrics


def compute_mrr_hit_k(pred: EvalPrediction):
    """
    This function computes the MRR and hit@k in the format of Huggingface's compute_metrics function, i.e.,
    Callable[[EvalPrediction], Dict].

    :param pred: given predictions.
    :return: dictionary string to metric values (mrr and hit_k).
    """
    global hit_k
    masked_indices = pred.label_ids != -100
    masked_words_num = np.count_nonzero(masked_indices)
    preds = pred.predictions[masked_indices]

    top_preds_indices = np.argsort(-preds)[:, :100]
    correct_word_indices = np.argwhere(pred.label_ids[masked_indices][:, np.newaxis] == top_preds_indices)[:, -1]
    mrr = np.sum(1 / np.add(correct_word_indices, 1)) / masked_words_num
    hit_k_res = np.count_nonzero(correct_word_indices < hit_k) / masked_words_num
    logging.info(f"MRR is {mrr}, hit_{hit_k} is {hit_k_res}")
    return {"mrr": mrr, f"hit_{hit_k}": hit_k_res}


def compute_mrr_and_all_hit_ks(pred: EvalPrediction):
    """
    This function computes the MRR and hit@k in the format of Huggingface's compute_metrics function, i.e.,
    Callable[[EvalPrediction], Dict].

    :param pred: given predictions.
    :return: dictionary string to metric values (mrr and hit_k).
    """
    global hit_k
    masked_indices = pred.label_ids != -100
    masked_words_num = np.count_nonzero(masked_indices)
    preds = pred.predictions[masked_indices]

    top_preds_indices = np.argsort(-preds)[:, :100]
    correct_word_indices = np.argwhere(pred.label_ids[masked_indices][:, np.newaxis] == top_preds_indices)[:, -1]
    mrr = np.sum(1 / np.add(correct_word_indices, 1)) / masked_words_num
    hit_k_res = [np.count_nonzero(correct_word_indices <= k) / masked_words_num for k in range(hit_k)]  # TODO: < or <=
    logging.info(f"MRR is {mrr}, hit_{hit_k} is {hit_k_res}")
    return {"mrr": mrr, f"hit_1 to hit_{hit_k}": hit_k_res}


def plot_metrics(hit_ks: List[float], k: int, mrrs_avgs: List[float], mrrs_stds: List[float], plot_name: str) -> None:
    """
    This function saves the plot of the MRRs and hit@ks for the different certainty levels.

    :param mrrs_stds:
    :param hit_ks: A list of the hit@ks for the different certainty levels
    :param k: The k of the hit_ks list
    :param mrrs_avgs: A list of the MRRs for the different certainty levels
    :param plot_name: A path to save the plot
    """
    plt.errorbar(range(len(mrrs_avgs)), mrrs_avgs, mrrs_stds, ecolor='r', elinewidth=1, capsize=3, marker='o',
                 label='mrrs')
    plt.plot(range(len(hit_ks)), hit_ks, label='hits')
    plt.legend()
    plt.title(f"The MRR and hit@{k} metrics by certainty level")
    plt.xlabel("Certainty values")
    plt.savefig(plot_name)
    plt.close()


def restore_original_text_from_jsons(id_text, project_name):
    """
    This function restores the original text for the JSONS given an id_text and a project name.
    :param id_text: A given id of a text
    :param project_name: A given name of a project
    :return: the raw text of the given text identifiers
    """
    project_path = glob.glob(f'{JSONS_DIR}/{project_name}', recursive=True)
    if not len(project_path):
        return ""
    rel_path = f'{project_path[0]}/corpusjson/{id_text}.json'
    if not os.path.isfile(rel_path):
        return ""
    doc_json = _load_json_from_path(rel_path)
    try:
        sents_dicts = doc_json['cdl'][0]['cdl'][-1]['cdl']
    except Exception as e:
        print(f"In file {rel_path} failed because of {e}")
        return None

    raw_text = get_raw_akk_text_from_json(sents_dicts)
    return raw_text


def predict_missing_signs(model: AutoModelForMaskedLM, tokenizer: PreTrainedTokenizer, input_path: str, k_beams: int,
                          output_path: str, nos_limit: int, sample_size: int = 0):
    """
    This function predicts missing signs in a given preprocessed ORACC input file.

    :param output_path:
    :param model: A given trained model
    :param tokenizer: A given tokenizer adjusted to the given model
    :param input_path: A given input file of the same format as the training input files
    :param k_beams: A given integer representing the number of beams to store in the evaluation of the multiple
    :param nos_limit: The max length sequence of missing signs to predict
    :param sample_size: The number of documents to randomly choose from the input. if sample = 0 -> take all documents.
    """
    # Get a sample of documents from the input file and augment them with several properties from the original jsons
    with open(input_path, "r", encoding='utf-8') as f_in:
        sample = random.sample(f_in.readlines(), sample_size) if sample_size > 0 else f_in.readlines()
        sample = add_properties_to_data(sample, ['genre', 'period', 'language', 'provenience'])

    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        top_k=k_beams,
    )

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for data_point in tqdm(sample, desc='lines loop'):
            data_point_json = json.loads(data_point)

            data_point_genre = data_point_json.get('genre')
            is_correct_genre = data_point_genre is not None and \
                               ('letter' in data_point_genre.lower() or 'royal inscription' in data_point_genre.lower())

            if not is_correct_genre or len(data_point_json['bert_input'].split()) == 0:
                continue

            data_point_json['missing_signs_num'] = data_point_json['bert_input'].count(MISSING_SIGN_CHAR)
            data_point_json['missing_signs_percent'] = data_point_json['missing_signs_num'] / len(
                data_point_json['bert_input'].split())

            encodings = tokenizer.encode(data_point_json['bert_input'])

            trunc_encodings = tokenizer.truncate_sequences(encodings, num_tokens_to_remove=len(encodings) - 450)[0]
            trunc_text = tokenizer.decode(trunc_encodings[1:-1])  # TODO: fix "." as delimiter

            try:
                pred_texts = predict_missing_signs_in_document(fill_mask=fill_mask, text=trunc_text, k_beams=k_beams,
                                                               mask_token=tokenizer.mask_token, nos_limit=nos_limit,
                                                               min_cont_window=10, text_lang=ORACCLanguage.Akkadian)
            except Exception as e:
                print(e)
                raise e

            final_json = dict()

            text = get_original_text(data_point_json)

            # Perform some basic preprocessing which has no semantic effect
            text = re.sub('[?*<>]', '', text)
            text = remove_squared_brackets(text, False)

            text = tokenizer.decode(tokenizer.encode(text)[1:-1])
            text_splitted = np.array(text.split())
            missing_sign_indices = np.where(text_splitted == 'x')[0]

            for pred_tokens, pred_probs, seq_indices in pred_texts:
                try:
                    cur_miss_bounds = missing_sign_indices[seq_indices[0]], missing_sign_indices[seq_indices[1]]
                    new_text = text.split()
                    for i in range(cur_miss_bounds[0], cur_miss_bounds[-1] + 1):
                        new_text[i] = MISSING_SIGN_IN_ANNOTATION  # same as '❌'
                    data_point_json['first_index'] = int(cur_miss_bounds[0])
                    data_point_json['missing_sequence_length'] = int(cur_miss_bounds[-1] - cur_miss_bounds[0] + 1)
                    data_point_json['hidden'] = list(zip(pred_tokens, pred_probs))
                    final_json['text'] = ' '.join(new_text)
                    final_json['meta'] = data_point_json
                    final_json['labels'] = list(pred_tokens)
                    json.dump(final_json, f_out)
                    f_out.write('\n')
                except Exception as e:
                    print(e)


def get_original_text(data_point_json):
    # Get the original text (before the preprocessing) from the jsons
    text = restore_original_text_from_jsons(data_point_json['id_text'], data_point_json['project_name'])
    if text:  # Couldn't find document in the jsons
        return text
    return get_raw_text_akk_from_html(data_point_json['id_text'], data_point_json['project_name'])


def predict_non_missing_signs(model: AutoModelForMaskedLM, tokenizer: PreTrainedTokenizer, input_path: str,
                              k_beams: int, hit_k: int, nos_seq_to_predict: int, sample_size: int = 0,
                              remove_hyphens=False):
    """


    :param remove_hyphens:
    :param output_path:
    :param model: A given trained model
    :param tokenizer: A given tokenizer adjusted to the given model
    :param input_path: A given input file of the same format as the training input files
    :param k_beams: A given integer representing the number of beams to store in the evaluation of the multiple
    :param nos_seq_to_predict: The number of consecutive missing signs to predict each time
    :param sample_size: The number of documents to randomly choose from the input. if sample = 0 -> take all documents.
    """

    # Get a sample of documents from the input file and augment them with several properties from the original jsons
    with open(input_path, 'r', encoding='utf-8') as f_in:
        sample = random.sample(f_in.readlines(), sample_size) if sample_size > 0 else f_in.readlines()
        # sample = add_properties_to_data(sample, ['genre', 'period', 'language', 'provenience'])

    fill_mask = pipeline(
        'fill-mask',
        model=model,
        tokenizer=tokenizer,
        top_k=k_beams,
    )

    mrr = Q = 0
    hit_ks = np.zeros(hit_k)
    for data_point in tqdm(sample, desc='lines loop'):
        data_point_json = json.loads(data_point)

        encodings = tokenizer.encode(data_point_json['bert_input'])

        trunc_encodings = tokenizer.truncate_sequences(encodings, num_tokens_to_remove=len(encodings) - 450)[0]
        trunc_text = tokenizer.decode(trunc_encodings[1:-1])  # TODO: fix "." as delimiter

        text_splitted_to_signs_and_seps = re.split('(\W+)', trunc_text)
        signs, seps = text_splitted_to_signs_and_seps[::2], text_splitted_to_signs_and_seps[1::2] + ['']
        assert len(signs) == len(seps)
        for start_ind in tqdm(range(len(signs) - nos_seq_to_predict)):
            end_ind = start_ind + nos_seq_to_predict
            new_signs = change_interval_in_list(signs, end_ind, start_ind, nos_seq_to_predict, MISSING_SIGN_CHAR)
            new_seps = change_interval_in_list(seps, end_ind, start_ind, nos_seq_to_predict, ' ')
            new_text = zigzag_two_lists(new_signs, new_seps)
            try:
                pred_res = predict_missing_signs_in_document(fill_mask=fill_mask, text=new_text, k_beams=k_beams,
                                                             mask_token=tokenizer.mask_token,
                                                             nos_limit=nos_seq_to_predict, min_cont_window=10,
                                                             text_lang=ORACCLanguage.Akkadian)[0]
            except Exception as e:
                print(e)
                continue
            if pred_res:
                if pred_res['missing_signs_occurences']['first'] == 0 and \
                        pred_res['missing_signs_occurences']['last'] == (nos_seq_to_predict - 1):
                    print(pred_res['missing_signs_occurences']['first'])
                    print(pred_res['missing_signs_occurences']['last'])
                    print(new_text)
                    print(trunc_text)
                    print(pred_res["preds"])

                preds_sorted = list(map(itemgetter(0), sorted(zip(pred_res["preds"], pred_res["probs"]),
                                                              key=itemgetter(1), reverse=True)))

                orig_masked_str = zigzag_two_lists(signs[start_ind: end_ind], seps[start_ind: end_ind]).strip()
                if orig_masked_str in preds_sorted:
                    correct_prediction_index = preds_sorted.index(orig_masked_str)
                    mrr += 1 / (correct_prediction_index + 1)
                    hit_ks[correct_prediction_index] += 1
                Q += 1

        for start_ind in range(len(hit_ks)):
            print(f"hit@{start_ind} is {sum(hit_ks[:start_ind + 1]) / Q}")
        print(f"MRR is {mrr / Q}")
        print(f"Q is {Q}")
    return mrr / Q, hit_ks / Q


def zigzag_two_lists(lst1: List[str], lst2: List[str]) -> str:
    return ''.join([sign + sep for sign, sep in zip(lst1, lst2)])


def change_interval_in_list(lst, end_ind, start_ind, interval_len, interval_char):
    return lst[:start_ind] + [interval_char] * interval_len + lst[end_ind:]


def signs_histogram(input_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    docs_counter = signs_counter = 0
    docs_dist = list()
    for line in lines:
        raw_text = json.loads(line)["bert_input"]
        signs_counter += len(re.split("(\W+)", raw_text))
        docs_counter += 1
        docs_dist.append(len(re.split("(\W+)", raw_text)))
    plt.hist(docs_dist, bins=range(0, 3000, 50))
    plt.title("Histogram of number of signs over documents")
    plt.ylabel("Number of documents")
    plt.xlabel("Number of signs")
    plt.show()


def sample_texts(input_file: str, sample_size: int, output_file: str, seed: int) -> None:
    # Get a sample of documents from the input file and augment them with several properties from the original jsons
    random.seed(seed)
    with open(input_file, 'r', encoding='utf-8') as f_in:
        sample = random.sample(f_in.readlines(), sample_size) if sample_size > 0 else f_in.readlines()
        # sample = add_properties_to_data(sample, ['genre', 'period', 'language', 'provenience'])
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(sample)


def predict_missing_signs_in_document(fill_mask: Pipeline, text: str, k_beams: int, mask_token: str, nos_limit: int,
                                      min_cont_window: int, text_lang):
    """
    This function predicts one sequence of missing signs in a given preprocessed text from ORACC.

    :param text_lang:
    :param fill_mask: A given transformers' filling mask pipeline
    :param text: A given string representing a preprocessed text from ORACC
    :param k_beams: A given integer representing the number of beams to store in the prediction
    :param mask_token: A given string representing a mask token in full_mask's tokenizer
    :return: The given text but with predictions of signs instead of missing signs
    :param nos_limit: The max length sequence of missing signs to predict
    """
    text = text.split()
    missing_sign_indices = np.where(np.array(text) == MISSING_SIGN_CHAR)[0]
    cons_miss_inds_list = np.split(missing_sign_indices, np.where(np.diff(missing_sign_indices) != 1)[0] + 1)

    for cons_miss_inds in cons_miss_inds_list:
        if cons_miss_inds.size == 0:
            return list()
        first_ind, last_ind = cons_miss_inds[0], cons_miss_inds[-1]

        if first_ind > 0 and text[first_ind - 1].endswith(MISSING_SIGN_CHAR):
            text[first_ind - 1] = text[first_ind - 1][:-1]
            text[first_ind] = ' '.join([MISSING_SIGN_CHAR, text[first_ind]])

        if last_ind < (len(text) - 1) and text[last_ind + 1].startswith(MISSING_SIGN_CHAR):
            text[last_ind + 1] = text[last_ind + 1][1:]
            text[last_ind] = ' '.join([text[last_ind], MISSING_SIGN_CHAR])

    pred_texts = list()
    for consec_indices in cons_miss_inds_list:  # tqdm(cons_miss_inds_list, desc='doc loop'):
        nos = sum(text[ind].count(MISSING_SIGN_CHAR) for ind in consec_indices)
        cont_window = text[consec_indices[0] - min_cont_window: consec_indices[0]] + \
                      text[consec_indices[-1] + 1: consec_indices[-1] + min_cont_window + 1]
        assert cont_window == text[consec_indices[0] - min_cont_window: consec_indices[0]] + \
               text[consec_indices[-1] + 1: consec_indices[-1] + min_cont_window + 1]
        full_context = True
        for non_sign_symbol in ['(', ')', MISSING_SIGN_CHAR, '...']:
            if any(non_sign_symbol in word for word in cont_window):
                full_context = False
                break
        if not full_context or nos > nos_limit:
            continue
        first_missing_idx = consec_indices[0]

        # Change prefix and post to include everything before and after the missing signs, accordingly
        prefix = ' '.join(text[:first_missing_idx])
        post = ' '.join(text[first_missing_idx + consec_indices.shape[0]:])
        if text_lang == ORACCLanguage.Akkadian:
            cur_best_k_preds, cur_best_k_probs = predict_k_beams(mask_token, prefix, post, nos, k_beams, fill_mask)
        elif text_lang == ORACCLanguage.English:
            cur_best_k_preds, cur_best_k_probs = predict_k_beams_english(mask_token, prefix, post, nos, k_beams,
                                                                         fill_mask)
        else:
            raise Exception(f"Illegal value for text_lang parameter: must be an ORACCLanguage instance")

        missing_sign_indices_list = list(missing_sign_indices)
        pred_texts.append(
            {

                "preds": cur_best_k_preds,
                "probs": cur_best_k_probs,
                "missing_signs_occurences":
                    {
                        "first": missing_sign_indices_list.index(consec_indices[0]),
                        "last": missing_sign_indices_list.index(consec_indices[-1])
                    }
            }
        )
    return pred_texts


def predict_tokens(fill_mask: Pipeline, text: str):
    """
    This function predicts tokens given a text with exactly one masked token.
    The number of tokens to predict is predefined in fill_mask.

    :param fill_mask: A given fill mask pipeline
    :param text: A given text with exactly one masked token
    :return: The predicted tokens and their corresponding probabilities
    """
    preds = fill_mask(text)
    new_tokens = [pred['token_str'] for pred in preds]
    probs = [pred['score'] for pred in preds]
    return new_tokens, probs


def predict_tokens_no_pipeline(model, tokenizer, text: str):
    """
    This function predicts tokens given a text with exactly one masked token.
    The number of tokens to predict is predefined in fill_mask.

    :param fill_mask: A given fill mask pipeline
    :param text: A given text with exactly one masked token
    :return: The predicted tokens and their corresponding probabilities
    """
    inputs = tokenizer(text)
    outputs = model(**inputs)
    predictions = outputs[0]
    sorted_preds, sorted_idx = predictions[0].sort(dim=-1, descending=True)
    for k in range(10):
        predicted_index = [sorted_idx[i, k].item() for i in range(0,24)]
        predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(1,24)]
        print(predicted_token)


def end_of_sign_found(token: str, preceding_token: str):
    """
    This function receives a token and its preceding token and returns whether that token ends an Akkadian sign.
    """
    if not preceding_token:
        return False
    if '-' in token or '.' in token:
        return True
    if not preceding_token.endswith('-') and not token.startswith('##'):
        return True
    return False


def end_of_word_found(token: str, preceding_token: str):
    """
    This function receives a token and its preceding token and returns whether that token ends an English word.
    """
    if not preceding_token:
        return False
    if '-' in token or '.' in token:
        return True
    if not preceding_token.endswith('-') and not token.startswith('##'):
        return True
    return False


def predict_k_beams(mask_token: str, prefix: str, post: str, nos: int, k: int, fill_mask: Pipeline):
    """
    This function predicts #nos signs placed between the prefix and the post using a k beams mechanism.

    :param mask_token: A given string representing a mask token in full_mask's tokenizer
    :param prefix: The begging a given text
    :param post: The end of a given text
    :param nos: The number of signs to predict between the prefix and the post
    :param k: The number of beams to store in the prediction process of a sign
    :param fill_mask: A given transformers' filling mask pipeline
    :return: #k best predictions of the #nos missing signs
    """
    pred_tokens, best_tokens = np.zeros((k, k), dtype=object), np.zeros(k, dtype=object)
    mat_probs, best_probs = np.ones((k, k)), np.ones(k)
    signs_counters, best_signs_counters = np.zeros((k, k), dtype=int), np.zeros(k, dtype=int)

    for i in range(k):
        best_tokens[i] = ""

    completed_predictions = 0
    while completed_predictions < k:
        # Generating k^2 candidates
        for i in range(k):
            if np.all(mat_probs[i] == 1) and i != 0:  # In first iteration we predict only the first row
                mat_probs[i] = 0
                continue

            if best_signs_counters[i] < nos:  # if not reached limit of predicted nos
                cur_text = generate_text_to_predict(prefix=prefix, post=post, mask_token=mask_token, nos=nos,
                                                    num_signs_predicted=best_signs_counters[i], cur_pred=best_tokens[i])
                new_tokens, new_probs = predict_tokens(fill_mask, cur_text)
                sign_adds = np.array([end_of_sign_found(new_token, best_tokens[i]) for new_token in new_tokens])
                signs_counters[i] = best_signs_counters[i] + sign_adds
                for j in range(k):
                    if signs_counters[i][j] >= nos and '-' not in new_tokens[j] and '.' not in new_tokens[j]:
                        new_tokens[j] = ""
                        new_probs[j] = int(new_tokens[j] not in new_tokens[:j])  # Zero out duplications in row
                    if new_tokens[j].startswith("##"):  # If continuation of a sign
                        new_tokens[j] = new_tokens[j][2:]
                    elif new_tokens[j]:  # If not empty string
                        new_tokens[j] = f" {new_tokens[j]}"

                # Update the (k x k) matrices with the new predictions
                pred_tokens[i] = np.array([best_tokens[i] + new_token for new_token in new_tokens])
                mat_probs[i] = best_probs[i] * np.array(new_probs)
            else:
                signs_counters[i] = best_signs_counters[i]
                pred_tokens[i][0] = best_tokens[i]
                mat_probs[i] = np.concatenate(([best_probs[i]], np.zeros(k - 1)))

        # Select k beams
        beams_inds = np.argpartition(mat_probs.flat, -k)[-k:]
        best_tokens = pred_tokens.flat[beams_inds]
        best_signs_counters = signs_counters.flat[beams_inds]
        best_probs = mat_probs.flat[beams_inds]

        completed_predictions = np.count_nonzero(best_signs_counters >= nos)

    return list(map(str.strip, best_tokens)), best_probs


def predict_k_beams_english(mask_token: str, prefix: str, post: str, now: int, k: int, fill_mask: Pipeline):
    """
    This function predicts #nos words placed between the prefix and the post using a k-beams mechanism.

    :param mask_token: A given string representing a mask token in full_mask's tokenizer
    :param prefix: The begging a given text
    :param post: The end of a given text
    :param now: The number of words to predict between the prefix and the post
    :param k: The number of beams to store in the prediction process of a word
    :param fill_mask: A given transformers' filling mask pipeline
    :return: #k best predictions of the #nos missing words
    """
    pred_tokens, best_tokens = np.zeros((k, k), dtype=object), np.zeros(k, dtype=object)
    mat_probs, best_probs = np.ones((k, k)), np.ones(k)
    words_counters, best_words_counters = np.zeros((k, k), dtype=int), np.zeros(k, dtype=int)

    for i in range(k):
        best_tokens[i] = ""

    completed_predictions = 0
    while completed_predictions < k:
        # Generating k^2 candidates
        for i in range(k):
            if np.all(mat_probs[i] == 1) and i != 0:  # In first iteration we predict only the first row
                mat_probs[i] = 0
                continue

            if best_words_counters[i] <= now:  # if not reached limit of predicted now
                cur_text = generate_text_to_predict(prefix=prefix, post=post, mask_token=mask_token, nos=now,
                                                    num_signs_predicted=best_words_counters[i], cur_pred=best_tokens[i])
                new_tokens, new_probs = predict_tokens(fill_mask, cur_text)
                words_counts_adds = np.array(
                    [not new_token.startswith('##') and new_token not in string.punctuation for new_token in
                     new_tokens])
                words_counters[i] = best_words_counters[i] + words_counts_adds
                for j in range(k):
                    if words_counters[i][j] > now and '-' not in new_tokens[j] and '.' not in new_tokens[j]:
                        new_tokens[j] = ""
                        new_probs[j] = int(new_tokens[j] not in new_tokens[:j])  # Zero out duplications in row
                    if new_tokens[j].startswith("##"):  # If continuation of a word
                        new_tokens[j] = new_tokens[j][2:]
                    elif new_tokens[j] and new_tokens[j] not in string.punctuation:  # If not empty string
                        new_tokens[j] = f" {new_tokens[j]}"

                # Update the (k x k) matrices with the new predictions
                pred_tokens[i] = np.array([best_tokens[i] + new_token for new_token in new_tokens])
                mat_probs[i] = best_probs[i] * np.array(new_probs)
            else:
                words_counters[i] = best_words_counters[i]
                pred_tokens[i][0] = best_tokens[i]
                mat_probs[i] = np.concatenate(([best_probs[i]], np.zeros(k - 1)))

        # Select k beams
        beams_inds = np.argpartition(mat_probs.flat, -k)[-k:]
        best_tokens = pred_tokens.flat[beams_inds]
        best_words_counters = words_counters.flat[beams_inds]
        best_probs = mat_probs.flat[beams_inds]

        completed_predictions = np.count_nonzero(best_words_counters > now)

    return list(map(str.strip, best_tokens)), best_probs


def generate_text_to_predict(prefix, post, mask_token, nos, num_signs_predicted=0, cur_pred=""):
    return prefix \
           + f"{cur_pred}" \
           + f" {mask_token} " \
           + f"{MISSING_SIGN_CHAR} " * (nos - num_signs_predicted - 1) \
           + post


def create_annotations_for_predictions(preds_file, signs_predicted_seq_len, signs_vocab, out_file, mask_seps):
    id_text2id = dict()

    with open(preds_file, 'r', encoding='utf-8') as f_in, open(out_file, 'w', encoding='utf-8') as f_out:
        for i, pred in enumerate(f_in):
            json_annotation = create_json_annotation(json.loads(pred), i, signs_vocab, signs_predicted_seq_len,
                                                     id_text2id,
                                                     mask_seps)
            if json_annotation:
                json.dump(json_annotation, f_out)
                f_out.write('\n')
    with open(f'{signs_predicted_seq_len}{"_no_seps" if not mask_seps else ""}_text2id.pickle', 'wb') as f_out:
        pickle.dump(id_text2id, f_out)


def create_json_annotation(pred_json, annotation_id, signs_vocab, num_signs_predicted, id_text2id, mask_seps) -> json:
    # Generate a version of the original text with minimal processing/changes
    orig_text = get_original_text(pred_json)
    orig_text_splitted = re.split('([\-. +]+)', orig_text)  # Split by Akkadian signs delimiters
    orig_text_signs, orig_text_seps = orig_text_splitted[::2], orig_text_splitted[1::2]

    orig_text_split_processed_signs, orig_text_split_processed_seps = list(), list()
    for sign, sep in zip(orig_text_signs, orig_text_seps):
        # Remove parts which are redundant semantically
        sign = _remove_redundant_parts(sign)
        sign = sign.replace(INTENTIONAL_HOLE_IN_CUNEIFORM, '')
        sign = re.sub('[?*<>]', '', sign)
        sign = remove_squared_brackets(sign, False)

        # Separate between signs and definers in curly brackets, the latters are in SUPERSCRIPTS_TO_UNICODE_CHARS
        for superscript in re.findall("({.*?})", sign):
            if superscript not in SUPERSCRIPTS_TO_UNICODE_CHARS:
                if superscript in {'{2}', '{lu₂}', '{munus}', '{f}', '{l}'}:  # If semantically important
                    id_text2id[pred_json['id_text']] = len(id_text2id)
                    return {}
                orig_text_split_processed_signs.append(superscript.upper())
                orig_text_split_processed_seps.append(' ')
                sign = sign.replace(superscript, '')

        if sign.strip():
            orig_text_split_processed_signs.append(sign)
            orig_text_split_processed_seps.append(sep)

    # Change missing sign emoji back to 'x' and then split preprocessed text by non-word delimiters
    text_split = re.split('(\W+)', pred_json['text'].replace(MISSING_SIGN_CHAR, 'x'))

    text_signs, text_seps = text_split[::2], text_split[1::2]
    first_masked_idx = text_signs.index('x')
    assert text_signs[first_masked_idx + num_signs_predicted - 1] == 'x' and text_signs[
        first_masked_idx + num_signs_predicted] != 'x'

    # Insert missing signs in the original text by their location in the preprocessed text
    orig_text_split_processed_signs[
    first_masked_idx: first_masked_idx + num_signs_predicted] = MISSING_SIGN_IN_ANNOTATION * num_signs_predicted
    # Change delimiters between the missing signs to spaces, similarly to "real" missing signs
    if mask_seps:
        orig_text_split_processed_seps[
        first_masked_idx: first_masked_idx + num_signs_predicted] = ' ' * num_signs_predicted
        gold = pred_json['label'].strip()
    else:
        gold = pred_json['label'].replace('-', '').replace(' ', '').strip()
    preds_sorted = list(map(itemgetter(0), sorted(zip(pred_json["preds"], pred_json["probs"]),
                                                  key=itemgetter(1), reverse=True)))
    labels = create_labels_for_annotation(preds_sorted, gold, generate_noise_token(signs_vocab))
    if pred_json['id_text'] not in id_text2id:
        id_text2id[pred_json['id_text']] = len(id_text2id)
    return {
        'meta': {
            'id_annotation': annotation_id,
            'gold_predicted': gold in pred_json['preds'],
            'language': pred_json['language'],
            'period': pred_json['period'],
            'provenience': pred_json['provenience'],
            'project_name': pred_json['project_name'],
            'genre': pred_json['genre'],
            'masked_indices': (first_masked_idx, first_masked_idx + num_signs_predicted - 1),
            'labels': labels,
            'text_id': id_text2id[pred_json['id_text']]
        },
        'text': zigzag_two_lists(orig_text_split_processed_signs, orig_text_split_processed_seps),
        'labels': labels
    }


def generate_noise_token(signs_vocab):
    return random.choice(signs_vocab)


def create_labels_for_annotation(model_preds, gold, noise):
    labels_for_annotation = [noise, gold]
    if gold in model_preds:
        model_preds.remove(gold)

    assert len(labels_for_annotation + model_preds[:3]) == 5  # TODO: magic numbers
    return labels_for_annotation + model_preds[:3]


def get_signs_vocab(input_file, num_of_signs, no_seps):
    # Generate a set of all #num_of_signs-tuples of signs appearing in a jsonl file containing texts in Akkadian
    signs_vocab = set()
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            text = json.loads(line)['bert_input']
            text_splitted = re.split(r'(\W+)', text)
            signs = text_splitted[::2]
            if no_seps:
                signs_vocab.update(signs)
            else:
                seps = text_splitted[1::2]
                pairs = [''.join(elem) for elem in zip(signs, seps)]
                temp = [''.join(pairs[i:i + num_of_signs]).strip() for i in range(len(pairs) - num_of_signs + 1)]
                signs_vocab.update(temp)

    return signs_vocab