import threading
import nltk
import os
import sys
import glob
import json
import re
from typing import Union
sys.path.append(os.path.abspath(os.curdir))
from preprocessing.main_preprocess import write_jsons_to_jsonl
IGNORED = {"project": [], "text": []}


def get_files(path: str, num=7, dirs: bool = False):
    files_list = [g for g in glob.glob(path)
                  if re.search(r'(P|Q|X)\d+.json$', g)] if not dirs else[g
                                                                         for g in glob.glob(path) if os.path.isdir(g) and "corpusjson" not in g]
    if num == 0:
        return files_list
    return get_files(f"{path}/*", num-1, dirs) + files_list


SUB_PROJECTS = {name.split("\\")[-1]: name.split("\\")[1]
                for name in get_files("jsons_unzipped", 7, True) if '\\' in name}

# def get_project(name: str):
#     return ()


def __from_json(data):
    '''HELPER: reads the json data, if the __From_JSON didn't work very well
    recursive function
    :param data: the data from the call. can be a list or a dictionary, and this will try to extract from it
    :type data: dict,list
    :return: a __from_json(data) value, if it has 'cdl' inside the dictionary, or in the dictionaries in the list. 
             if it has no 'cdl' value, it returns a list of the words from the json
    :rtype: dict, list
    '''
    if isinstance(data, list):
        for pos in data:
            if "cdl" in data:
                return __from_json(pos)
        return data
    elif isinstance(data, dict):
        return __from_json(data["cdl"])


def __From_JSON(file):
    '''HELPER: reads the json data as a dictionary and tries to return the list of elements

    :param file: json file that was open
    :type file: dict
    :return: list of the elements in the json, if it works, otherwise, tries __from_json
    :rtype: list, __from_json
    '''
    try:
        return file['cdl'][0]['cdl'][-1]['cdl'][0]['cdl']
    except KeyError:
        return __from_json(file)
    except IndexError:
        return __from_json(file)


def _get_data(file):
    '''HELPER: reads the json data and return its word list

    :param file: a json file path to read
    :type file: str
    :return: the list of elements from the json file
    :rtype: list
    '''
    try:
        with open(file, "r", encoding="utf_8") as file:
            Json = __From_JSON(json.load(file))
        return Json
    except json.JSONDecodeError:
        return {}


def __join_words(phrase: str):
    return " ".join([p[p.find("$")+1:] for p in phrase.split("&")])


def _get_word(data: dict):
    if data.get("sig"):
        word = data["sig"]
        return __join_words(word) if '&' in word else word[word.find('$')+1:]
    else:
        return data["f"]["form"] if data.get("f") and data["f"].get("form") else ""


def _remove_phrases(data: dict):
    try:
        if data["node"] == "l":
            return _get_word(data)
        elif data["node"] == "c":
            return _get_word(data["cdl"][0])
        else:
            return ""
    except KeyError:
        return ""
    except IndexError:
        return ""


def read_file(filename: str):
    file = filename.split("\\")
    text_id = file[-1].split(".")[0]
    if text_id not in IGNORED["text"]:
        file_dict = {"id_text": text_id, "sub_project_name": file[file.index("corpusjson")-1]}
        se = [_remove_phrases(d) for d in _get_data(filename) if d and _remove_phrases(d)]
        try:
            file_dict["raw_text"] = " ".join(se)
            file_dict["raw_text"]
            file_dict["raw_text"] = re.sub(r'\$[A-z]+', 'x', file_dict["raw_text"])
            file_dict["raw_text"] = re.sub(r'\s[o]\s', ' ° ', file_dict["raw_text"])
        except IndexError:
            pass
        finally:
            return file_dict
    else:
        return {}


def read_project(project_name: str, default_dir: str = "jsons_unzipped"):
    if project_name not in IGNORED["project"] and project_name in os.listdir("jsons_unzipped") and not re.search(
            r"\d$", project_name):
        return [read_file(text) for text in get_files(f'{default_dir}/{project_name}')]
    elif project_name in SUB_PROJECTS:
        files = [read_file(text) for text in get_files(f'{default_dir}/{SUB_PROJECTS[project_name]}')
                 if project_name in text]
        return files
    return []


def write_jsonl(project_name: str, reading_dir: str = "jsons_unzipped", writing_dir: str = "data/jsonl"):
    if not os.path.exists(writing_dir):
        os.mkdir(writing_dir)
    project = read_project(project_name, reading_dir)
    write_jsons_to_jsonl(project, f'{writing_dir}/{project_name}.jsonl', "utf_8", False)


if __name__ == "__main__":
    # # s = read_file(get_files("jsons_unzipped")[10])
    # print(get_files("jsons_unzipped")[-1])
    # write_jsonl("akklove")
    for p in os.listdir("jsons_unzipped"):
        write_jsonl(p)
