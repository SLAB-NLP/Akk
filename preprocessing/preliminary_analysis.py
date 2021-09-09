import argparse
import json
from typing import List
import requests
from bs4 import BeautifulSoup
import pandas as pd

import zipfile
from glob import glob
import os
from pathlib import Path
import re
from collections import Counter

from tqdm import tqdm

ENGLISH_JSONL_FILE = "data/english_from_website.jsonl"

DATA_JSONL_FILENAME = "data_from_jsons.jsonl"

JSONS_DIR = "jsons_unzipped"


def _load_json_to_dict(json_path):
    with open(json_path, encoding='utf-8') as json_file:
        if os.stat(json_path).st_size != 0:  # If file is not empty:
            return json.load(json_file)


def _calc_num_of_lines(json_dict):
    counter = 0
    for d in json_dict:
        if d.get('type') == 'line-start':
            counter += 1
    return counter


def _calc_num_of_words(json_dict):
    return sum([1 if d.get('frag') else 0 for d in json_dict])


def _count_words_by_language_html(soup):
    """
    For a given bs object representing a text, returns a counter of all the words in the text by language/dialect.
    """
    counter = Counter()
    words = soup.find_all("a", class_="cbd")
    for word in words:
        counter.update([re.match(r".*(akk-x-.*):", word.attrs['href']).group(1)])
    return counter


def _get_raw_text_html(soup):
    words = soup.find_all("a", class_="cbd")
    return ' '.join([word.text for word in words])


def _calc_num_of_words_by_language(json_dict):
    c = Counter()
    for d in json_dict:
        if 'f' in d:
            c.update([d['f']['lang']])
    return c


def _add_words_to_vocab(json_dict, vocab):
    for d in json_dict:
        if d.get('frag'):
            vocab.add(d['frag'])


def json_to_csv(json_dict, csv_filename, index=None):
    df = pd.DataFrame(json_dict, index=index)
    df.to_csv(csv_filename)


def update_dict(d, k1, k2, add=1):
    d.setdefault(k1, dict())
    d[k1][k2] = d[k1].get(k2, 0) + add


def calc_sentences():
    """
    Old function that calculates and writes the number of complete and incomplete sentences in ORACC.
    An incomplete sentence is considered as a sentence with missing signs/parts.
    """
    incomplete_sents_counter_by_project, incomplete_sents_counter_general = dict(), dict()
    complete_sents_counter_by_project, complete_sents_counter_general = dict(), dict()
    sents_counter_by_project, sents_counter_general = dict(), dict()

    for directory in os.listdir(JSONS_DIR):
        incomplete_sents_counter_by_project[directory] = dict()
        complete_sents_counter_by_project[directory] = dict()
        sents_counter_by_project[directory] = dict()

        for path in Path(os.path.join(JSONS_DIR, directory)).rglob('catalogue.json'):
            if str(path) == "jsons_unzipped\cams\catalogue.json":  # multiple projects as subdirectories
                continue
            d = _load_json_to_dict(str(path))
            if d and d.get('members'):
                for member in d.get('members').values():
                    lang, period = member.get('language', 'unspecified'), member.get('period', 'unspecified')
                    id_text = member.get('id_text', "") + member.get('id_composite', "")
                    html_dir = "/".join(path.parts[1:-1])
                    url = f"http://oracc.iaas.upenn.edu/{html_dir}/{id_text}/html"
                    print(url)
                    res = requests.get(url)
                    soup = BeautifulSoup(res.text, "html.parser")
                    results = soup.find_all("span", {"class": "cell"})
                    for result in results:
                        is_complete = True
                        for content in result.contents:
                            if isinstance(content, str):
                                if "..." in content.strip():  # if words are missing
                                    is_complete = False
                                elif content.strip() in [".", "?", "!"]:
                                    if is_complete:
                                        update_dict(complete_sents_counter_general, period, lang)
                                        update_dict(complete_sents_counter_by_project[directory], period, lang)
                                    else:
                                        update_dict(incomplete_sents_counter_general, period, lang)
                                        update_dict(incomplete_sents_counter_by_project[directory], period, lang)
                                    update_dict(sents_counter_general, period, lang)
                                    update_dict(sents_counter_by_project[directory], period, lang)
                                    is_complete = True

                    print(incomplete_sents_counter_by_project)
                    print(incomplete_sents_counter_general)

        outdir = './results'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        json_to_csv(complete_sents_counter_general, os.path.join(outdir, "complete_sentences_general.csv"))
        json_to_csv(incomplete_sents_counter_general, os.path.join(outdir, "incomplete_sentences_general.csv"))
        json_to_csv(sents_counter_general, os.path.join(outdir, "sentences_general.csv"))
        for proj_data in sents_counter_by_project:
            json_to_csv(incomplete_sents_counter_by_project[proj_data],
                        os.path.join(outdir, f"incomplete_sentences_{proj_data}.csv"))
            json_to_csv(complete_sents_counter_by_project[proj_data],
                        os.path.join(outdir, f"complete_sentences_{proj_data}.csv"))
            json_to_csv(sents_counter_by_project[proj_data], os.path.join(outdir, f"sentences_{proj_data}.csv"))


def calc_words():
    """
    Old function that calculates and writes the number of words in ORACC.
    An incomplete sentence is considered as a sentence with missing signs/parts.
    """
    words_counter_general, words_counter_by_project = dict(), dict()

    for directory in os.listdir(JSONS_DIR):
        if directory == "qcat":  # Shitty project
            continue
        words_counter_by_project[directory] = dict()
        for path in Path(os.path.join(JSONS_DIR, directory)).rglob('catalogue.json'):
            if str(path) == os.path.join(JSONS_DIR, "cams", "catalogue.json"):  # Waste of time
                continue
            texts_json = _load_json_to_dict(str(path))
            if texts_json and texts_json.get('members'):
                for member in texts_json.get('members').values():
                    id_text = member.get('id_text', "") + member.get('id_composite', "")
                    json_file_path = os.path.join(path.parent, "corpusjson", f'{id_text}.json')
                    if os.path.isfile(json_file_path):
                        d = _load_json_to_dict(json_file_path)
                        try:
                            json_dict = d['cdl'][0]['cdl'][-1]['cdl'][0]['cdl']
                            num_of_words = _calc_num_of_words(json_dict)
                            # num_of_lines = calc_num_of_lines(json_dict)
                        except Exception as e:
                            print(e)
                            continue
                    else:
                        html_dir = "/".join(path.parts[1:-1])
                        url = f'http://oracc.iaas.upenn.edu/{html_dir}/{id_text}/html'
                        res = requests.get(url)
                        soup = BeautifulSoup(res.text, "html.parser")
                        num_of_words = len(soup.find_all("a", class_="cbd"))

                    if num_of_words > 0:
                        lang, period = member.get('language', 'unspecified'), member.get('period', 'unspecified')
                        update_dict(words_counter_general, period, lang, num_of_words)
                        update_dict(words_counter_by_project[directory], period, lang, num_of_words)
                        print(directory, id_text)
                        print(words_counter_general)

    outdir = './results'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    json_to_csv(words_counter_general, os.path.join(outdir, "words_count.csv"))
    for proj_data in words_counter_by_project:
        json_to_csv(words_counter_by_project[proj_data], os.path.join(outdir, f"words_count_{proj_data}.csv"))


def get_words_counts_of_proj(proj_dirname="saao"):
    words_counter_general, words_counter_by_project = Counter(), dict()
    jsons_proj_dir_path = os.path.join(JSONS_DIR, proj_dirname)
    for path in Path(jsons_proj_dir_path).rglob('catalogue.json'):
        if str(path) in [os.path.join(jsons_proj_dir_path, "catalogue.json"),
                         os.path.join(jsons_proj_dir_path, "knpp", "catalogue.json")]:  # Waste of time
            continue
        sub_proj_name = path.parts[2]
        words_counter_by_project[sub_proj_name] = Counter()
        texts_json = _load_json_to_dict(str(path))
        if texts_json and texts_json.get('members'):
            for member in texts_json.get('members').values():
                id_text = member.get('id_text', "") + member.get('id_composite', "")
                json_file_path = os.path.join(path.parent, "corpusjson", f'{id_text}.json')
                cur_counter = Counter()
                if os.path.isfile(json_file_path):  # If file exists in the jsons
                    d = _load_json_to_dict(json_file_path)
                    try:
                        if d:
                            json_dict = d['cdl'][0]['cdl'][-1]['cdl'][0]['cdl']
                            cur_counter = _calc_num_of_words_by_language(json_dict)
                    except Exception as e:
                        print(e)
                        continue
                else:  # If the file doesn''t exist in the jsons -> look for it online
                    html_dir = "/".join(path.parts[1:-1])
                    url = f'http://oracc.iaas.upenn.edu/{html_dir}/{id_text}/html'
                    res = requests.get(url)
                    soup = BeautifulSoup(res.text, "html.parser")
                    cur_counter = _count_words_by_language_html(soup)

                if len(cur_counter):  # If the link contains words and not empty
                    words_counter_general.update(cur_counter)
                    words_counter_by_project[sub_proj_name].update(cur_counter)

    outdir = './results_saa'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    json_to_csv(words_counter_general, os.path.join(outdir, "words_count.csv"), index=["#words"])
    json_to_csv(words_counter_by_project, os.path.join(outdir, "words_count_by_project.csv"))


def count_translated_files_in_proj(proj_dirname="saao"):
    count_translated_files, count_translated_files_by_subproj = 0, dict()
    count_not_translated_files, count_not_translated_files_by_subproj = 0, dict()
    jsons_proj_dir_path = os.path.join(JSONS_DIR, proj_dirname)
    for path in Path(jsons_proj_dir_path).rglob('catalogue.json'):
        if str(path) in [os.path.join(jsons_proj_dir_path, "catalogue.json"),
                         os.path.join(jsons_proj_dir_path, "knpp", "catalogue.json")]:  # Waste of time
            continue
        subproj_name = path.parts[2]
        count_translated_files_by_subproj[subproj_name] = 0
        count_not_translated_files_by_subproj[subproj_name] = 0
        texts_json = _load_json_to_dict(str(path))
        if texts_json and texts_json.get('members'):
            for member in texts_json.get('members').values():
                id_text = member.get('id_text', "") + member.get('id_composite', "")
                html_dir = "/".join(path.parts[1:-1])
                url = f'http://oracc.iaas.upenn.edu/{html_dir}/{id_text}/html'
                res = requests.get(url)
                soup = BeautifulSoup(res.text, "html.parser")
                is_translated = bool(soup.find_all("a", {"name": "a.P224485_project-en"}))
                count_translated_files += is_translated
                count_translated_files_by_subproj[subproj_name] += is_translated
                count_translated_files += not is_translated
                count_translated_files_by_subproj[subproj_name] += not is_translated
                print(count_translated_files)
                print(count_not_translated_files)

    outdir = './results_saa'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    json_to_csv(count_translated_files, os.path.join(outdir, "num_translated.csv"), index=["#words"])
    json_to_csv(count_translated_files_by_subproj, os.path.join(outdir, "num_translated_by_project.csv"))
    json_to_csv(count_translated_files, os.path.join(outdir, "num_not_translated.csv"), index=["#words"])
    json_to_csv(count_translated_files_by_subproj, os.path.join(outdir, "num_not_translated_by_project.csv"))


def find_vocabulary():
    vocabs = dict()

    for directory in os.listdir(JSONS_DIR):
        for path in Path(os.path.join(JSONS_DIR, directory)).rglob('catalogue.json'):
            if str(path) == os.path.join(JSONS_DIR, "cams", "catalogue.json"):  # multiple projects as subdirectories
                continue
            texts_json = _load_json_to_dict(str(path))
            if texts_json and texts_json.get('members'):
                for member in texts_json.get('members').values():
                    lang = member.get('language', 'unspecified')
                    vocabs.setdefault(lang, set())
                    id_text = member.get('id_text', "") + member.get('id_composite', "")
                    html_dir = "/".join(path.parts[1:-1])
                    url = f'http://oracc.iaas.upenn.edu/{html_dir}/{id_text}/html'
                    res = requests.get(url)
                    soup = BeautifulSoup(res.text, "html.parser")
                    for tag in soup.find_all("a", class_="cbd"):
                        vocabs[lang].add(re.sub('[\[\]⸢⸣]', '', tag.text))
    with open("vocab.json", "w+") as output_file:
        json.dump(vocabs, output_file)


def count_documents():
    documents_counter, documents_counter_by_proj = dict(), dict()

    for directory in os.listdir(JSONS_DIR):
        if directory == "qcat":  # Shitty project
            continue
        documents_counter_by_proj[directory] = dict()
        for path in Path(os.path.join(JSONS_DIR, directory)).rglob('catalogue.json'):
            if str(path) == os.path.join(JSONS_DIR, "cams", "catalogue.json"):  # Waste of time
                continue
            texts_json = _load_json_to_dict(str(path))
            if texts_json and texts_json.get('members'):
                for member in texts_json.get('members').values():
                    id_text = member.get('id_text', "") + member.get('id_composite', "")
                    json_file_path = os.path.join(path.parent, "corpusjson", f'{id_text}.json')
                    if os.path.isfile(json_file_path):
                        d = _load_json_to_dict(json_file_path)
                        try:
                            json_dict = d['cdl'][0]['cdl'][-1]['cdl'][0]['cdl']
                            is_valid_doc = _calc_num_of_words(json_dict) > 0
                        except Exception as e:
                            print(e)
                            continue
                    else:
                        html_dir = "/".join(path.parts[1:-1])
                        url = f'http://oracc.iaas.upenn.edu/{html_dir}/{id_text}/html'
                        res = requests.get(url)
                        soup = BeautifulSoup(res.text, "html.parser")
                        is_valid_doc = len(soup.find_all("a", class_="cbd")) > 0

                    if is_valid_doc:
                        lang, period = member.get('language', 'unspecified'), member.get('period', 'unspecified')
                        update_dict(documents_counter, period, lang)
                        update_dict(documents_counter_by_proj[directory], period, lang)

    outdir = './results'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    json_to_csv(documents_counter, os.path.join(outdir, "documents_count.csv"))
    for proj_data in documents_counter_by_proj:
        json_to_csv(documents_counter_by_proj[proj_data], os.path.join(outdir, f"documents_count_{proj_data}.csv"))


def genres_histogram(input_files: List[str], output_file: str) -> None:
    """
    This function calculates a histogram of number of documents and words per genre in ORACC.
    The histogram is written to a JSONL file.

    :param input_files: A list of JSONL files where each line is a json representing a text in ORACC
    :param output_file: A JSONL file to write the histogram

    """
    documents_hist, words_hist = Counter(), Counter()
    for raw_akk_json_file in tqdm(input_files):
        with open(raw_akk_json_file, 'r', encoding='utf-8') as fIn:
            for line in fIn:
                cur_json = json.loads(line)
                cur_genre = cur_json.get('genre', 'unspecified')
                cur_words_num = len(cur_json['bert_input'].split())
                if cur_words_num > 0:
                    documents_hist.update([cur_genre])
                    words_hist[cur_genre] += cur_words_num

    with open(output_file, 'w') as fOut:
        json.dump(documents_hist.most_common(50), fOut)
        fOut.write('\n')
        json.dump(words_hist.most_common(50), fOut)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate M-BERT model on Akkadian MLM task")
    parser.add_argument('--raw_akk_jsonl_file',
                        help='A list of JSONL files comma-delimited, containing the raw jsons of ORACC',
                        default=JSONS_DIR, )
    parser.add_argument('--histogram_file_path',
                        help='A path to write the histogram of words and documents per genre')

    args = parser.parse_args()

    if args.histogram_file_path:
        genres_histogram(args.raw_akk_jsonl_files, args.histogram_file_path)
