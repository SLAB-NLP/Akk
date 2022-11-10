from argparse import ArgumentParser
import io
import json
import logging
import zipfile
from collections import Counter
from typing import Dict, Set
import requests
from bs4 import BeautifulSoup
import os
import re
import glob
import requests
from pathlib import Path
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_dir)

JSONS_DIR = "jsons"

SUB_PROJECTS_IN_NEO_ASS = ["01", "05", "06", "07", "09", "11", "12", "14", "15", "16"]

CORPUS_DIRNAME = "corpusjson"


def _load_json_from_path(json_path: str) -> Dict:
    """
    This helper function loads a json from a given path, with the exception of empty files.

    :param json_path: A given string representing a path to a json file.
    :return: The json file (if it is a valid json file).
    """
    with open(json_path, "r", encoding='utf-8') as json_file:
        if os.stat(json_path).st_size != 0:  # If the file is not empty:
            return json.load(json_file)


def get_raw_english_texts_of_project(project_dirname: str, output_file, seen_id_texts, catalogue) -> Set[str]:
    """
    This function parses the raw texts of a project in ORACC translated to English and writes a list of jsons
    containing the raw texts of the given project and basic metadata.

    :param catalogue: A dictionary of dictionaries of the metadata of all texts in the project
    :param project_dirname: A given string representing the path to the project's directory.
    :param output_file: An output file for the jsons of the raw texts.
    :param seen_id_texts: A set of id texts that were already seen, to prevent duplicity.
    :return: The set of seen id texts
    """
    all_paths = glob.glob(f'{project_dirname}/**/corpusjson/*.json', recursive=True)
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for filename in tqdm(all_paths, desc=f'English_files_{project_dirname}'):
            cur_json = _load_json_from_path(filename)
            if not cur_json:
                continue
            project_name = cur_json.get('project')
            if not project_name:
                logging.info(f'Could not find id_text for {filename}')
            id_text = cur_json['textid']
            text_properties = catalogue.get(id_text)
            if id_text in seen_id_texts:
                logging.debug(f'Found duplication of {project_name}/{id_text} in English')
                continue
            url = f"http://oracc.iaas.upenn.edu/{project_name}/{id_text}/html"
            logging.info(url)
            res = requests.get(url)
            soup = BeautifulSoup(res.text, "html.parser")
            results = soup.find_all("span", {"class": "cell"})
            raw_text = " ".join(["".join([content if isinstance(content, str) else content.text
                                          for content in result.contents]) for result in results])
            raw_text = raw_text.replace('\n', ' ')
            if raw_text:
                seen_id_texts.add(id_text)
                json.dump({
                    "id_text": id_text,
                    "project_name": project_name,
                    "provenience": text_properties.get('provenience'),
                    "genre": text_properties.get('genre'),
                    'period': text_properties.get('period'),
                    'url': url,
                    'language': text_properties.get('language'),
                    "raw_text": raw_text,
                },
                    f_out,
                )
                f_out.write('\n')

    return seen_id_texts


def num_words_in_english(jsonl_file):
    words_counter = 0
    with open(jsonl_file, "r", encoding='utf-8') as f_in:
        for line in f_in:
            cur_json = json.loads(line)
            if cur_json["project_name"].startswith("saa"):
                cur_json["raw_text"] = re.sub(r'\([^)]*\)', '', cur_json["raw_text"])
                words_counter += len(cur_json["raw_text"].split())
    print(words_counter)


def get_raw_text_akk_from_html(id_text, project_name):
    """

    :param id_text:
    :param project_name:
    :return:
    """
    url = f'http://oracc.iaas.upenn.edu/{project_name}/{id_text}/html'
    res = requests.get(url)
    if res.status_code != 200:
        print("******STATUS CODE IS NOT 200***********")
        return ""
    soup = BeautifulSoup(res.text, "html.parser")
    raw_text = _get_raw_text_html(soup)
    return raw_text


def _get_raw_text_html(soup):
    tags = soup.find_all('span', class_=lambda value: value and value.startswith('w '))
    signs = list()
    for tag in tags:
        temp_tag = tag.find('a')
        if temp_tag:
            tag = temp_tag
        for sign in tag.contents:
            if isinstance(sign, str):
                signs.append(sign)
            elif sign.name == 'span':
                signs.append(sign.text)
            elif sign.name == 'sup':
                signs.append("{" + sign.text + "}")
    return ' '.join(signs)


def get_raw_akk_texts_of_project(project_dirname: str, output_file: str, seen_id_texts: set,
                                 total_lang_counter: Counter, catalogue: dict) -> [Set[str], Counter]:
    """
    This function parses the raw texts of a project in ORACC and writes a list of jsons containing the raw texts of
    the given project and basic metadata.

    :param catalogue:
    :param project_dirname: A given string representing the path to the project's directory.
    :param output_file: An output file for the jsons of the raw texts.
    :param seen_id_texts: A set of id texts that were already seen, to prevent duplicity.
    :param total_lang_counter: A counter of all words (or signs?) in all the corpus.
    :return: The set of seen id texts.
    """
    all_paths = glob.glob(f'{project_dirname}/**/corpusjson/*.json', recursive=True)
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for filename in tqdm(all_paths, desc=f'Akkadian_files_{project_dirname}'):
            cur_json = _load_json_from_path(filename)
            try:
                id_text = cur_json['textid']
                text_properties = catalogue[id_text]
                if id_text in seen_id_texts:
                    logging.debug(f'Found duplication of {project_name}/{id_text} in Akkadian')
                    continue
                project_name = cur_json['project']
                sents_dicts = cur_json['cdl'][0]['cdl'][-1]['cdl']
                raw_text = get_raw_akk_text_from_json(sents_dicts)
                text_lang_counter = _get_lang_dist_of_raw_text(sents_dicts, Counter())
            except Exception as e:
                print(f"In file {filename} failed because of {e}")
                continue

            # Remove texts which mostly contains Sumerian word (and not Akkadian)
            if text_lang_counter['sux'] + text_lang_counter['sux-x-emesal'] > 0.5 * sum(text_lang_counter.values()):
                logging.debug(f'{project_name}/{id_text}:{text_lang_counter}')
                continue

            logging.debug(f'{project_name}/{id_text}:{text_lang_counter}')
            if raw_text:
                seen_id_texts.add(id_text)
                total_lang_counter.update(text_lang_counter)
                json.dump({
                    "id_text": id_text,
                    "project_name": project_name,
                    "provenience": text_properties.get('provenience'),
                    "genre": text_properties.get('genre'),
                    'period': text_properties.get('period'),
                    'language': text_properties.get('language'),
                    'url': f"http://oracc.iaas.upenn.edu/{project_name}/{cur_json['textid']}/html",
                    "raw_text": raw_text,
                },
                    f_out,
                )
                f_out.write('\n')

    return seen_id_texts, total_lang_counter


def get_raw_akk_text_from_json(sents_dicts):
    return " ".join([_get_raw_text(sent_dict['cdl']) for sent_dict in sents_dicts if _is_sent(sent_dict)])


def _get_raw_text(json_dict: dict) -> str:
    """
    This function gets the raw text of a given transliterated tablet in ORACC recursively.
    It appends each instance in the tablet only once (even if there are multiple possible meanings).

    :param json_dict: A given dictionary representing some portion of the words in the tablet.
    :return: The aforementioned raw text.
    """
    previous_ref: str = ""
    raw_texts = list()
    for d in json_dict:
        if _is_sent(d):  # If node represents a sentence -> call recursively to the inner dictionary
            raw_texts.extend(_get_raw_text(d['cdl']).split())
        elif _is_word(d):  # If node represents a word
            if previous_ref != d.get('ref'):  # If encountered new instance:
                cur_text = d['frag'] if d.get('frag') else d['f']['form']
                raw_texts.append(cur_text + _get_addition(d))
                previous_ref = d.get('ref')

    return " ".join(raw_texts)


def _get_lang_dist_of_raw_text(json_dict: dict, lang_counter) -> Counter:
    """
    This function gets the raw text of a given transliterated tablet in ORACC recursively.
    It appends each instance in the tablet only once (even if there are multiple possible meanings).

    :param json_dict: A given dictionary representing some portion of the words in the tablet.
    :return: The aforementioned raw text.
    """
    previous_ref: str = ""
    for d in json_dict:
        if _is_sent(d):  # If node represents a sentence -> call recursively to the inner dictionary
            lang_counter = _get_lang_dist_of_raw_text(d['cdl'], lang_counter)
        elif _is_word(d):  # If node represents a word
            if previous_ref != d.get('ref'):  # If encountered new instance:
                lang_counter.update([d['f']['lang']])
                previous_ref = d.get('ref')

    return lang_counter


def _is_sent(d: Dict) -> bool:
    return d.get('node') == 'c'


def _is_word(d: Dict) -> bool:
    return d.get('node') == 'l'


def _get_addition(d: Dict) -> str:
    """
    This function looks for an asterisk or a question mark in a dictionary representing a word in a tablet from ORACC.

    :param d: A given dictionary as described above.
    :return An asterisk or a question mark if one of the word's signs has one, otherwise an empty string.
    """
    has_signs_dicts = 'f' in d and 'gdl' in d.get('f')
    if has_signs_dicts:
        for sign_dict in d['f']['gdl']:
            if 'gdl_collated' in sign_dict:  # If cur sign has an asterisk
                return "*"
            if 'queried' in sign_dict:  # If cur sign has a question mark
                return "?"
    return ""


def download_projects_jsons(output_dir: str = JSONS_DIR):
    """
    This function downloads ORACC projects as json files from its website: 'http://oracc.org'..

    :param output_dir: A path to a directory to save the projects jsons.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    res = requests.get('http://oracc.museum.upenn.edu/projects.json')
    projects_list = json.loads(res.content)['public']
    for project_name in tqdm(projects_list, desc='projects loop'):
        project_json_url = f'http://oracc.org/{project_name}/json'
        r = requests.get(project_json_url, stream=True)
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
                zip_ref.extractall(output_dir)
        except zipfile.BadZipFile as e:
            print(f'{e}:{project_name}')


if __name__ == '__main__':
    parser = ArgumentParser('Scraping the data from the website of ORACC')
    parser.add_argument('--output_dir',
                        help='A path to a directory to save the projects jsons from the website',
                        default=JSONS_DIR,
                        )
    args = parser.parse_args()

    download_projects_jsons(args.output_dir)
