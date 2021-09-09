import glob
import pickle
from pathlib import Path

import argparse
import json
import re
from collections import Counter
from enum import Enum
import sys
import unicodedata
from typing import List, Dict, Tuple
import logging
import os
from sklearn.model_selection import train_test_split
import nltk
from tqdm import tqdm

SEED = 2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from preprocessing.data_dist import get_all_catalogues_members
from akkadian_bert.utils import text_language
from preprocessing.pse_words import decide_pse_word
from preprocessing.scraping import get_raw_akk_texts_of_project, get_raw_english_texts_of_project, JSONS_DIR, \
    get_raw_akk_text_from_json, _load_json_from_path

INTENTIONAL_HOLE_IN_CUNEIFORM = 'o'

FREQ_WORD_THRESHOLD = 3

LOGGING_FILE = "preprocessing_log.log"

PRIVATE_USE_CATEGORY = "Co"

# IMPORTANT: You MUST have "jsons_unzipped" directory (contact Koren Lazar for details: koren.lzr@gmail.com)
# to run the pipeline.

AKK_JSONL_FILE = "data/raw/akk_jsons.jsonl"
ENG_JSONL_FILE = "data/raw/eng_jsons.jsonl"

MISSING_SIGN_CHAR = "\U0001F607"

SUPERSCRIPTS_TO_UNICODE_CHARS = {
    "{1}": "\U0001F600",
    "{m}": "\U0001F600",
    "{d}": "\U0001F601",
    "{f}": "\U0001F602",
    "{MÍ}": "\U0001F602",
    "{ki}": "\U0001F603",
    "{kur}": "\U0001F604",
    "{giš}": "\U0001F605",
    "{uru}": "\U0001F606",
    "{V}": "",
}
SUPERSCRIPTS_CONTENTS = ["1", "m", "d", "f", "MÍ", "ki", "kur", "giš", "uru", "V"]

SUBSCRIPTS_LIST = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]

# Initializing the unicode characters list used for replacing signs with subscripts
unicode_list = list()
for c in map(chr, range(sys.maxunicode + 1)):
    if unicodedata.category(c) == PRIVATE_USE_CATEGORY:
        unicode_list.append(c)

subscript_dict: Dict[str, str] = dict()


class Certainty(Enum):
    """
    This Enum class represents the certainty level of a given Neo-Assyrian word in a tablet.
    """
    SURE, FORGOTTEN_SIGN, FIXED_BY_EDITOR, HAS_DOUBTS, BLURRED, MISSING_BUT_COMPLETED, MISSING = range(7)


def remove_curly_brackets(word: str, remove_superscripts: bool = True) -> str:
    """
    This function removes curly brackets in a given word.
    In case the brackets contain a superscript, then the superscript is replaced by a fixed unicode character.
    Otherwise, the content inside the curly brackets is capitalized and a "." is added as a suffix.

    :param remove_superscripts:
    :param word: A given string representing a word.
    :return: The given word after removing the curly brackets.
    """
    if "{" in word and "}" in word:
        for superscript in SUPERSCRIPTS_TO_UNICODE_CHARS:  # All superscripts are wrapped in curly brackets
            sub_chars = '' if remove_superscripts else SUPERSCRIPTS_TO_UNICODE_CHARS[superscript]  # TODO: delete line?
            word = word.replace(superscript.lower(), sub_chars)
            word = word.replace(superscript.upper(), sub_chars)

        # If the word has curly brackets that do not represent a superscript:
        word = re.sub("{(.*?)}", lambda pat: pat.group(1).upper() + ".", word)  # Remove curly brackets and add a "."
    return word


def replace_subscripts(word: str) -> str:
    """
    This function replaces signs with subscripts with some unicode character.
    The purpose of this replacement is to represent better signs that differ by subscripts.

    :param word: A given string representing a word.
    :return: The given word after replacing.
    """
    global subscript_dict
    signs = list()
    has_subscript = False
    for sign in re.split(r"(\W)", word):  # Splitting the word while keeping the delimiters
        if any(subscript in sign for subscript in SUBSCRIPTS_LIST):  # If current sign has subscripts:
            has_subscript = True
            if sign not in subscript_dict:  # If not added yet, find a suitable unicode as a value
                subscript_dict[sign] = unicode_list[len(subscript_dict)]
            sign = subscript_dict[sign]
        signs.append(sign)
    return "".join(signs) if has_subscript else word


def remove_subscripts(word: str) -> str:
    """
    This function removes all subscripts from a given word.

    :param word: A given word
    :return: The given word stripped from subscripts.
    """
    rx = '[' + re.escape(''.join(SUBSCRIPTS_LIST)) + ']'
    return re.sub(rx, '', word)


def remove_superscripts(word: str) -> str:
    """
    This function removes all superscripts from a given word.

    :param word: A given word
    :return: The given word stripped from superscripts.
    """
    rx = '[' + re.escape(''.join(SUPERSCRIPTS_TO_UNICODE_CHARS.keys())) + ']'
    return re.sub(rx, '', word)


def remove_squared_brackets(word: str, remove_missing: bool) -> str:
    """
    This function omits full/upper squared brackets in a given word.
    If the word contains a missing part (represented by "x" or "..."), then the function omits all of it.

    :param remove_missing: A boolean flag representing whether to remove missing parts from the texts
    :param word: A given string representing a word.
    :return: The given word after omitting.
    """
    if remove_missing and _has_missing_parts(word):
        return ""
    word = re.sub("\\[?(.*?)]?", r"\1", word)  # Remove full squared brackets
    return re.sub("⸢?(.*?)⸣?", r"\1", word)  # Remove upper squared brackets


def _has_missing_parts(word):
    return "x" in word or "..." in word or MISSING_SIGN_CHAR in word


def preprocess_text_akkadian(text: str, remove_hyphens: bool, freq_dist: nltk.FreqDist, remove_missing, remove_subs,
                             remove_supers) -> (List[dict]):
    """
    This function preprocess a given text in Akkadian.

    :param remove_supers: A boolean flag representing whether to remove all superscripts
    :param remove_subs: A boolean flag representing whether to remove all subscripts
    :param remove_missing: A boolean flag representing whether to remove missing parts from the texts
    :param text: A string representing a document
    :param remove_hyphens: A boolean flag representing whether to remove hyphens from words
    :param freq_dist: An nltk.FreqDist of the words in the training data.
    :return: A list of triples, each containing the original word, the preprocessed word and its certainty level.
    """
    preprocessed_triplets: List[dict] = list()
    encountered_full_squared_brackets = encountered_upper_squared_brackets = False
    for word in text.split():
        original_word = word
        word = re.sub(r'\+', '-', word)  # TODO: should we do that?
        word = _remove_redundant_parts(word)
        certainty_level, encountered_full_squared_brackets, encountered_upper_squared_brackets = \
            certainty_hierarchy(word, encountered_full_squared_brackets, encountered_upper_squared_brackets)
        word = word.replace('x', MISSING_SIGN_CHAR)
        word = word.replace(INTENTIONAL_HOLE_IN_CUNEIFORM, '')
        word = re.sub('[?*<>]', '', word)  # This characters are redundant after determining certainty level
        word = remove_squared_brackets(word, remove_missing)
        # word = remove_subscripts(word) if remove_subs else replace_subscripts(word)  # TODO: delete line?
        word = remove_subscripts(word)
        if freq_dist and freq_dist[word] < FREQ_WORD_THRESHOLD:  # Convert only scarce words
            word = decide_pse_word(word)
        word = remove_curly_brackets(word, remove_supers)
        if remove_hyphens:
            word = word.replace('-', '')

        if word.strip():  # Do not add empty strings
            preprocessed_triplets.append({"original": original_word,
                                          "preprocessed": word,
                                          "certainty": certainty_level.name,
                                          })
    return preprocessed_triplets


def preprocess_text_english(text: str, remove_missing):
    text = text.replace('[', '').replace(']', '')  # Remove redundant squared parenthesis
    text = re.sub("\\(([^)]*)\\)", lambda pat: '' if has_digit(pat.group(1)) else pat.group(1), text)
    if remove_missing:
        text = re.sub(r'\.\.\.+', '', text)
    return ' '.join(text.split())


def has_digit(s):
    return any(char.isdigit() for char in s)


def get_data_freq_dist(input_file: str, remove_missing, remove_subs,
                       remove_hyphens: bool = False) -> nltk.FreqDist:
    """
    This function computes the frequency distribution of a given input jsonl file which contains a raw_data field in
    every line.

    :param remove_subs: A boolean flag representing whether to remove all subscripts
    :param remove_missing: A boolean flag representing whether to remove missing parts from the texts
    :param input_file: A path to a given input file
    :param remove_hyphens: A boolean flag representing whether to remove hyphens from words
    :return: The aforementioned frequency distribution.
    """
    freq_dist = nltk.FreqDist()
    with open(input_file, "r", encoding='utf-8') as f_in:
        for line_in in f_in:
            cur_json = json.loads(line_in)
            for word in cur_json["raw_text"].split():
                word = _remove_redundant_parts(word)
                word = re.sub("[?*<>]", "", word)
                word = remove_squared_brackets(word, remove_missing=remove_missing)
                word = remove_subscripts(word)  # if remove_subs else replace_subscripts(word)
                if remove_hyphens:
                    word = word.replace('-', '')
                if word:
                    freq_dist.update([word])
    return freq_dist


def certainty_hierarchy(word: str, in_full_squared_brackets: bool, in_upper_squared_brackets: bool) \
        -> Tuple[Certainty, int, int]:
    """
    This function finds the certainty level of a given word according to a hierarchy of the Certainty Enum class.

    :param word: A given word.
    :param in_full_squared_brackets: A counter of the full squared brackets (to know if we are inside one).
    :param in_upper_squared_brackets: A counter of the upper squared bracket (to know if we are inside one).
    :return: The certainty level of the word in the hierarchy and the two updated flags.
    """
    if _has_missing_parts(word):
        certainty = Certainty.MISSING
    elif in_full_squared_brackets or "[" in word or "]" in word:
        certainty = Certainty.MISSING_BUT_COMPLETED
    elif in_upper_squared_brackets or "⸢" in word or "⸣" in word:
        certainty = Certainty.BLURRED
    elif "?" in word:
        certainty = Certainty.HAS_DOUBTS
    elif "*" in word:
        certainty = Certainty.FIXED_BY_EDITOR
    elif "<" in word or ">" in word:
        certainty = Certainty.FORGOTTEN_SIGN
    else:
        certainty = Certainty.SURE
    if word.find("[") != -1 or word.find("]") != -1:
        in_full_squared_brackets = word.rfind("[") > word.rfind("]")
    if word.find("⸢") != -1 or word.find("⸣") != -1:
        in_upper_squared_brackets = word.rfind("⸢") > word.rfind("⸣")

    # We can't allow negative values in the counts (which do occur as the editors are imperfect human beings)
    return certainty, in_full_squared_brackets, in_upper_squared_brackets


def preprocess_akk_file(input_filename: str, remove_hyphens: bool, pseudo_words: bool, remove_missing: bool,
                        remove_subs: bool, remove_supers: bool) -> List[Dict]:
    """
    This function preprocess all documents in a given jsonl file (each line is a json representing a document).

    :param remove_supers: A boolean flag representing whether to remove all superscripts
    :param remove_subs: A boolean flag representing whether to remove all subscripts
    :param input_filename: A given jsonl filename.
    :param remove_hyphens: A boolean flag representing whether to remove hyphens from words.
    :param pseudo_words: A boolean flag representing whether to use pseudo words
    :param remove_missing: A boolean flag representing whether to remove missing parts from the texts
    :return A list of preprocessed jsons
    """
    data_jsons = list()

    freq_dist = get_data_freq_dist(input_file=input_filename,
                                   remove_missing=remove_missing,
                                   remove_subs=remove_subs,
                                   remove_hyphens=remove_hyphens) if pseudo_words else None

    with open(input_filename, "r", encoding='utf-8') as f_in:
        for line_in in tqdm(f_in.readlines(), desc='Akkadian preprocessing loop'):
            cur_json = json.loads(line_in)
            cur_json["preprocessed_words"] = preprocess_text_akkadian(text=cur_json["raw_text"],
                                                                      remove_hyphens=remove_hyphens,
                                                                      freq_dist=freq_dist,
                                                                      remove_missing=remove_missing,
                                                                      remove_subs=remove_subs,
                                                                      remove_supers=remove_supers)
            cur_json["bert_input"] = " ".join([w["preprocessed"] for w in cur_json["preprocessed_words"]])
            data_jsons.append(cur_json)
    return data_jsons


def preprocess_eng_file(input_file, remove_missing) -> List[Dict]:
    data_jsons = list()
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            input_json = json.loads(line)
            bert_input = preprocess_text_english(input_json["raw_text"], remove_missing=remove_missing)
            data_jsons.append({
                **input_json,
                "bert_input": bert_input,
                "lang": "eng",
            })
    return data_jsons


def write_jsons_to_jsonl(jsons_list: List[Dict], filename: str, encoding: str = 'utf-8', ensure_ascii: bool = False):
    """
    This function writes a given list of jsons to a jsonl file.

    :param jsons_list: A given list of jsons
    :param filename: A path to a jsonl file to write the jsons list to
    :param encoding: A string representing the encoding of the output jsonl file
    :param ensure_ascii: A boolean flag representing whether to ensure ascii characters in the jsonl file
    """
    with open(filename, "w", encoding=encoding) as f_out:
        for cur_json in jsons_list:
            json.dump(cur_json, f_out, ensure_ascii=ensure_ascii)
            f_out.write('\n')


def _remove_redundant_parts(word: str) -> str:
    """
    This function removes redundant parts in a given Neo-Assyrian word in ORACC.

    :param word: A given word
    :return: The given word after removing redundant parts
    """
    word = re.sub(r"lu₂[a-z]*", "lu₂", word)
    return re.sub(r"@?v|\\[a-z]*", "", word)  # Redundant part


def restore_original_text_from_jsons(id_text: str, project_name: str, text_lang: text_language, jsons_dir):
    """
    This function restores the original text for the JSONS given an id_text and a project name.
    :param jsons_dir:
    :param text_lang: The language of the text
    :param id_text: A given id of a text
    :param project_name: A given name of a project
    :return: the raw text of the given text identifiers
    """
    project_path = glob.glob(f'{jsons_dir}/{project_name}', recursive=True)
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


def add_properties_to_data(preprocessed_data_jsonl: List[str], properties: List[str]):
    """
    This function extends each row of a preprocessed jsonl file with the given properties from the original jsonl files.
    :param preprocessed_data_jsonl: A given preprocessed jsonl file
    :param properties: A list of properties to add to each row
    :return:
    """
    new_jsons = list()
    all_members = get_all_catalogues_members()
    for data in preprocessed_data_jsonl:
        try:
            cur_data_json = json.loads(data)
            if cur_data_json['id_text'] in all_members:
                cur_member_json = all_members[cur_data_json['id_text']]
                for prop in properties:
                    prop_val = cur_member_json.get(prop)
                    cur_data_json[prop] = prop_val
                project_name = cur_member_json.get('project')
                cur_data_json['project_name'] = project_name
                cur_data_json['url'] = f'http://oracc.iaas.upenn.edu/{project_name}/{cur_data_json["id_text"]}'
                logging.info(cur_data_json['url'])
            new_jsons.append(json.dumps(cur_data_json))
        except Exception as e:
            print(e)
    return new_jsons


def preprocess_raw_eng_data_file(raw_eng_data_file, output_train_file, output_test_file, seed):
    """
    This function receives an English jsonl data file and preprocess it.

    :param raw_eng_data_file: A path to an English raw data file
    :param output_train_file: A path to the train preprocessed data file
    :param output_test_file: A path to the test preprocessed data file
    """
    preprocessed_eng_data = preprocess_eng_file(raw_eng_data_file, remove_missing=False)
    train_eng_preprocessed_data, test_eng_preprocessed_data = train_test_split(
        preprocessed_eng_data,
        test_size=0.2,
        random_state=seed,
    )
    write_jsons_to_jsonl(train_eng_preprocessed_data, output_train_file)
    write_jsons_to_jsonl(test_eng_preprocessed_data, output_test_file)


def preprocess_raw_akk_data_file(raw_akk_data_file, output_train_file, output_test_file, seed):
    """
    This function receives an Akkadian jsonl data file and preprocess it.

    :param raw_akk_data_file: A path to an Akkadian raw data file
    :param output_train_file: A path to the train preprocessed data file
    :param output_test_file: A path to the test preprocessed data file
    """
    preprocessed_akk_data = preprocess_akk_file(
        input_filename=raw_akk_data_file,
        remove_missing=False,
        remove_hyphens=False,
        pseudo_words=False,
        remove_subs=True,
        remove_supers=True, )
    train_akk_preprocessed_data, test_akk_preprocessed_data = train_test_split(
        preprocessed_akk_data,
        test_size=0.2,
        random_state=seed,
    )
    write_jsons_to_jsonl(train_akk_preprocessed_data, output_train_file)
    write_jsons_to_jsonl(test_akk_preprocessed_data, output_test_file)


def extend_preprocessed_file(preprocessed_file_input, preprocessed_file_output, keys_to_add: List[str]):
    """
    This function receives a jsonl file as an input with the keys "id_text" and "sub_project_name",
    and extends each json according to the given keys to add from the original jsonl files.

    :param preprocessed_file_input: A path for a preprocessed English jsonl file
    :param preprocessed_file_output: A path to write the extended preprocessed English jsonl file
    :param keys_to_add: A list of properties to add to each json.
    :return:
    """
    with open(preprocessed_file_input, 'r', encoding='utf-8') as f_in:
        preprocessed_eng_file_jsons = f_in.readlines()
    extended_jsonl = add_properties_to_data(preprocessed_eng_file_jsons, keys_to_add)
    with open(preprocessed_file_output, 'w', encoding='utf-8') as f_out:
        for preprocessed_json in extended_jsonl:
            json.dump(json.loads(preprocessed_json), f_out)
            f_out.write('\n')


def initialize_output_file(filename):
    file_path = Path(filename)
    Path(file_path.parent).mkdir(parents=True, exist_ok=True)

    open(filename, "w").close()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data from ORACC")
    parser.add_argument('--project_names',
                        help="The directory name of the chosen project",
                        default="all")
    parser.add_argument('--jsons_dir',
                        help='The path to the directory containing the ORACC texts jsons',
                        default=JSONS_DIR)
    parser.add_argument('--log_file',
                        help="Path for the logging file",
                        default=LOGGING_FILE)
    parser.add_argument('--raw_akk_data_file',
                        help="Path for the raw Akkadian text file",
                        default=AKK_JSONL_FILE)
    parser.add_argument('--raw_eng_data_file',
                        help="Path for the raw English data file",
                        default=ENG_JSONL_FILE)
    parser.add_argument('--raw_extended_akk_data_file',
                        help='Path for the raw extended Akkadian data file')
    parser.add_argument('--raw_extended_eng_data_file',
                        help='Path for the raw extended English data file')
    parser.add_argument('--do_scraping',
                        help='If True, raw text is generated from jsons/website',
                        action='store_true')
    parser.add_argument('--do_preprocessing',
                        help='If True, preprocessing phase is done',
                        action='store_true')
    parser.add_argument('--do_extended_english_preprocessing',
                        help='A boolean flag representing whether to run extended preprocessing to English',
                        action='store_true')
    parser.add_argument('--do_extended_akkadian_preprocessing',
                        help='A boolean flag representing whether to run extended preprocessing to English',
                        action='store_true')
    parser.add_argument('--remove_hyphens',
                        help='If True, hyphens are removed from words (eventually)',
                        action='store_true')
    parser.add_argument('--use_pseudo_words',
                        help='If True, some words are replaced by pseudo-words, according to certain patterns',
                        action='store_true')
    parser.add_argument('--remove_missing',
                        help='If True, missing parts are removed from text',
                        action='store_true')
    parser.add_argument('--remove_subscripts',
                        help='If True, subscripts are removed from text',
                        action='store_true')
    parser.add_argument('--remove_superscripts',
                        help='If True, superscripts are removed from text',
                        action='store_true')
    parser.add_argument('--exclude_english',
                        help='Boolean representing whether to exclude English translations of ORACC',
                        action='store_true')
    parser.add_argument('--train_akk_preprocessed_data_file',
                        help='Path for the Akkadian train preprocessed data file',
                        default='data/preprocessed/akk_train.jsonl')
    parser.add_argument('--test_akk_preprocessed_data_file',
                        help='Path for the Akkadian test preprocessed data file',
                        default='data/preprocessed/akk_test.jsonl')
    parser.add_argument('--train_eng_preprocessed_data_file',
                        help='Path for the English train preprocessed data file',
                        default='data/preprocessed/eng_train.jsonl')
    parser.add_argument('--test_eng_preprocessed_data_file',
                        help='Path for the English test preprocessed data file',
                        default='data/preprocessed/eng_test.jsonl')

    args = parser.parse_args()

    initialize_output_file(args.log_file)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename=args.log_file, mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(name)s %(message)s'))
    logger.addHandler(handler)

    if args.do_scraping:
        seen_id_texts_akk, seen_id_texts_eng = set(), set()
        projects_dirnames = [dp for (dp, dn, filenames) in os.walk('jsons') if
                             'corpusjson' in dn and os.listdir(os.path.join(dp, 'corpusjson'))]
        initialize_output_file(args.raw_akk_data_file)
        if not args.exclude_english:
            initialize_output_file(args.raw_eng_data_file)
        lang_counter = Counter()
        for proj_dirname in tqdm(projects_dirnames, desc='projects loop'):
            catalogue_filename = os.path.join(proj_dirname, 'catalogue.json')
            with open(catalogue_filename) as f_catalogue:
                catalogue = json.load(f_catalogue).get('members')
            seen_id_texts_akk, lang_counter = get_raw_akk_texts_of_project(
                project_dirname=proj_dirname,
                output_file=args.raw_akk_data_file,
                seen_id_texts=seen_id_texts_akk,
                total_lang_counter=lang_counter,
                catalogue=catalogue)
            if not args.exclude_english:
                seen_id_texts_eng = get_raw_english_texts_of_project(project_dirname=proj_dirname,
                                                                     output_file=args.raw_eng_data_file,
                                                                     seen_id_texts=seen_id_texts_eng,
                                                                     catalogue=catalogue)
        print(lang_counter)

    if args.do_preprocessing:
        initialize_output_file(args.train_akk_preprocessed_data_file)
        initialize_output_file(args.test_akk_preprocessed_data_file)

        preprocessed_akk_data = preprocess_akk_file(input_filename=args.raw_akk_data_file,
                                                    remove_hyphens=args.remove_hyphens,
                                                    pseudo_words=args.use_pseudo_words,
                                                    remove_missing=args.remove_missing,
                                                    remove_subs=args.remove_subscripts,
                                                    remove_supers=args.remove_superscripts)
        train_akk_preprocessed_data, test_akk_preprocessed_data = train_test_split(
            preprocessed_akk_data,
            test_size=0.2,
            random_state=SEED)
        write_jsons_to_jsonl(train_akk_preprocessed_data, args.train_akk_preprocessed_data_file)
        write_jsons_to_jsonl(test_akk_preprocessed_data, args.test_akk_preprocessed_data_file)

        if not args.exclude_english:
            initialize_output_file(args.train_eng_preprocessed_data_file)
            initialize_output_file(args.test_eng_preprocessed_data_file)
            preprocessed_eng_data = preprocess_eng_file(args.raw_eng_data_file, args.remove_missing)
            train_eng_preprocessed_data, test_eng_preprocessed_data = train_test_split(
                preprocessed_eng_data,
                test_size=0.2,
                random_state=SEED)
            write_jsons_to_jsonl(train_eng_preprocessed_data, args.train_eng_preprocessed_data_file)
            write_jsons_to_jsonl(test_eng_preprocessed_data, args.test_eng_preprocessed_data_file)

    if args.do_extended_english_preprocessing:
        # Running the English pipeline after scraping
        logging.info("Started extending English raw data files")
        extend_preprocessed_file(args.raw_eng_data_file, args.raw_extended_data_file,
                                 ["genre", "period", "language", "provenience", "project_name", "url"])
        logging.info("Finished extending English raw data files")
        logging.info("Started preprocessing extended English raw data files")
        preprocess_raw_eng_data_file(
            raw_eng_data_file=args.raw_extended_data_file,
            output_train_file=args.train_eng_preprocessed_data_file,
            output_test_file=args.test_eng_preprocessed_data_file,
            seed=SEED
        )
        logging.info("Finished preprocessing extended English raw data files")

    if args.do_extended_akkadian_preprocessing:
        # Extending preprocessed files with extra metadata
        logging.info("Started extending Akkadian raw data files")
        extend_preprocessed_file(args.raw_akk_data_file, args.raw_extended_akk_data_file,
                                 ["genre", "period", "language", "provenience", "project_name", "url"])
        logging.info("Finished extending Akkadian raw data files")
        logging.info("Started preprocessing extended Akkadian raw data files")
        preprocess_raw_akk_data_file(
            raw_akk_data_file=args.raw_extended_akk_data_file,
            output_train_file=args.train_akk_preprocessed_data_file,
            output_test_file=args.test_akk_preprocessed_data_file,
            seed=SEED
        )
        logging.info("Finished preprocessing extended Akkadian raw data files")


if __name__ == '__main__':
    main()
