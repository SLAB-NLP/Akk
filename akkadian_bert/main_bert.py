import argparse
import json
import logging
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from preprocessing.main_preprocess import write_jsons_to_jsonl
from akkadian_bert.evaluate_bert import (
    eval_compute_metrics,
    update_hit_k,
)
from akkadian_bert.train_bert import MAIN_TOKENIZER_DIRNAME, bert_from_scratch, m_bert
from akkadian_bert.utils import natural_number


def main():
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate M-BERT model on Akkadian MLM task")
    parser.add_argument('--model_dir',
                        help="The directory name of the model",
                        default="tiny_bert",
                        )
    parser.add_argument('--log_file',
                        help="Path for the logging file",
                        default='temp_log.log'
                        # type=argparse.FileType('r'),
                        )
    parser.add_argument('--preprocessed_akk_train_json_file',
                        help='The path of the Akkadian preprocessed train file',
                        default='data/preprocessed/akk_train.jsonl')
    parser.add_argument('--preprocessed_akk_test_json_file',
                        help="The path of the Akkadian preprocessed test file",
                        default='data/preprocessed/akk_test.jsonl',
                        # type=argparse.FileType('r'),
                        )
    parser.add_argument('--preprocessed_eng_train_json_file',
                        help="The path of the English preprocessed train file",
                        default='data/preprocessed/eng_train.jsonl',
                        # type=argparse.FileType('r'),
                        )
    parser.add_argument('--preprocessed_eng_test_json_file',
                        help="The path of the English preprocessed test file",
                        default='data/preprocessed/eng_test.jsonl'
                        # type=argparse.FileType('r'),
                        )
    parser.add_argument('--plot_file',
                        help="Path for the plot of the mrr and hit@k",
                        # type=argparse.FileType('w'),
                        )
    parser.add_argument('--hit_k',
                        help="k value for the hit@k metric",
                        type=natural_number,
                        default=5,
                        )
    parser.add_argument('--epochs',
                        help="The number of epochs to train the model",
                        type=natural_number,
                        default=30,
                        )
    parser.add_argument('--do_train',
                        help="Boolean flag to call the main train function",
                        action='store_true',
                        )
    parser.add_argument('--do_eval',
                        help="Boolean flag to call the automatic evaluation function",
                        action='store_true',
                        )
    parser.add_argument('--do_eval_by_genres',
                        help="Boolean flag to calculate the metrics as a function of genres",
                        action='store_true',
                        )

    parser.add_argument('--do_manual_eval_akkadian',
                        help="Boolean flag to create manual evaluation file of data in Akkadian",
                        action='store_true',
                        )
    parser.add_argument('--do_manual_eval_english',
                        help="Boolean flag to create manual evaluation file of data in English",
                        action='store_true',
                        )

    parser.add_argument('--batch_size',
                        # required=True,
                        type=natural_number,
                        default=2,
                        )
    parser.add_argument('--k_beams',
                        type=natural_number,
                        default=5,
                        )
    parser.add_argument('--nos_sequence_limit',
                        type=natural_number,
                        default=5,
                        )
    parser.add_argument('--from_scratch',
                        help='Boolean flag to train bert from scratch',
                        action='store_true'
                        )
    parser.add_argument('--sample_size',
                        type=int,
                        )
    parser.add_argument('--tokenizer_subdir',
                        default=MAIN_TOKENIZER_DIRNAME)
    parser.add_argument('--predict_signs_only',
                        action='store_true'
                        )
    parser.add_argument('--manual_input_file',
                        help='Path of the input file for the manual evaluation',
                        )
    parser.add_argument('--masked_texts_path',
                        help='Path of the input file for the manual evaluation, after masking',
                        )
    parser.add_argument('--predictions_path',
                        help='Path of the model\'s predictions for the manual evaluation',
                        )
    parser.add_argument('--shorten',
                        help='Number of docs to evaluate on',
                        default=0,
                        type=int,
                        )
    parser.add_argument('--genres_to_include',
                        help='A list of genres delimited by comma',
                        )
    parser.add_argument('--metrics_file',
                        help='A path to a pickle file to save the evaluation metrics')
    parser.add_argument('--include_english', action='store_true',
                        help='Boolean repreenting whether to include English in training')

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    update_hit_k(args.hit_k)
    if args.model_dir:
        logging.info(f'The model directory is {args.model_dir}')
    if args.do_train:
        if args.from_scratch:
            logging.info("Starting 'BERT from scratch' training")
            bert_from_scratch(
                preprocessed_akk_train_file=args.preprocessed_akk_train_json_file,
                preprocessed_eng_train_file=args.preprocessed_eng_train_json_file if args.include_english else None,
                preprocessed_akk_test_file=args.preprocessed_akk_test_json_file,
                model_dir=args.model_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        else:
            logging.info("Starting M-BERT training")
            m_bert(
                preprocessed_akk_train_file=args.preprocessed_akk_train_json_file,
                preprocessed_eng_train_file=args.preprocessed_eng_train_json_file if args.include_english else None,
                preprocessed_akk_test_file=args.preprocessed_akk_test_json_file,
                model_dir=args.model_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

    if args.do_eval:
        seen_id_texts = set()
        with open(args.preprocessed_akk_train_json_file, 'r', encoding='utf-8') as f_train:
            for train_akk_doc in f_train:
                seen_id_texts.add(json.loads(train_akk_doc)['id_text'])
        unseen_test_docs = []
        seen_test_docs = []
        with open(args.preprocessed_akk_test_json_file, 'r', encoding='utf-8') as f_test:
            genres_to_include = None if not args.genres_to_include else args.genres_to_include.split(',')
            for test_akk_doc in f_test:
                test_akk_doc_json = json.loads(test_akk_doc)
                if genres_to_include and test_akk_doc_json['genre'].lower() not in genres_to_include:
                    continue
                if test_akk_doc_json['id_text'] not in seen_id_texts:
                    unseen_test_docs.append(test_akk_doc_json)
                    seen_id_texts.add(test_akk_doc_json['id_text'])
                else:
                    seen_test_docs.append(test_akk_doc_json)
        logging.info(f'Size of filtered Akkadian test dataset is {len(unseen_test_docs)}')
        logging.info(f'Size of seen Akkadian test dataset is {len(seen_test_docs)}')
        filtered_akk_test_file = 'data/filtered_test_akk.jsonl'
        write_jsons_to_jsonl(unseen_test_docs, filtered_akk_test_file)

        metrics_dict = eval_compute_metrics(preprocessed_akk_input_file=filtered_akk_test_file,
                                            model_dir=args.model_dir,
                                            batch_size=args.batch_size, shorten=args.shorten,
                                            tokenizer_subdir=args.tokenizer_subdir)
        with open(args.metrics_file, 'wb') as handle:
            pickle.dump(metrics_dict, handle)

    if args.do_eval_by_genres:
        logging.info("Performing automatic evaluation by genres")
        logging.info(f"The model's dir is {args.model_dir}")
        seen_id_texts = set()
        with open(args.preprocessed_akk_train_json_file, 'r', encoding='utf-8') as f_train:
            for train_akk_doc in f_train:
                seen_id_texts.add(json.loads(train_akk_doc)['id_text'])
        unseen_test_docs = []
        seen_test_docs = []
        if args.genres_to_include:
            genres_to_include = args.genres_to_include.split(',')
            logging.info(f'The genres to include are {genres_to_include}')
            docs_by_genres_dict = {genre: [] for genre in genres_to_include}
        else:
            logging.info("Including all genres in the evaluation")
            genres_to_include = dict()
            docs_by_genres_dict = dict()
        with open(args.preprocessed_akk_test_json_file, 'r', encoding='utf-8') as f_test:
            for test_akk_doc in f_test:
                test_akk_doc_json = json.loads(test_akk_doc)
                doc_genre = test_akk_doc_json['genre']
                if args.genres_to_include and doc_genre.lower() not in genres_to_include:
                    continue
                if test_akk_doc_json['id_text'] not in seen_id_texts:
                    unseen_test_docs.append(test_akk_doc_json)
                    seen_id_texts.add(test_akk_doc_json['id_text'])
                    if args.genres_to_include:
                        docs_by_genres_dict[doc_genre].append(test_akk_doc_json)
                    else:
                        if doc_genre not in docs_by_genres_dict:
                            docs_by_genres_dict[doc_genre] = list()
                        docs_by_genres_dict[doc_genre].append(test_akk_doc_json)
                else:
                    seen_test_docs.append(test_akk_doc_json)
        logging.info(f'Size of filtered Akkadian test dataset is {len(unseen_test_docs)}')
        logging.info(f'Size of seen Akkadian test dataset is {len(seen_test_docs)}')
        metrics_dict = {}
        temp_file = 'temp.jsonl'
        for genre in docs_by_genres_dict.keys():
            write_jsons_to_jsonl(docs_by_genres_dict[genre], temp_file)
            logging.info(f"Calculating metrics for genre:{genre}")
            metrics_dict[genre] = eval_compute_metrics(
                preprocessed_akk_input_file=temp_file,
                model_dir=args.model_dir,
                batch_size=args.batch_size,
                shorten=args.shorten,
                tokenizer_subdir=args.tokenizer_subdir)
        with open(args.metrics_file, 'wb') as handle:
            pickle.dump(metrics_dict, handle)


if __name__ == '__main__':
    main()
