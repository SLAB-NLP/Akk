import threading
import json
import glob
import sys
import os
sys.path.append(os.path.abspath(os.curdir))

LANG_LIST = []


def __from_json(data):
    '''HELPER: reads the json data, if the __From_JSON didn't work very well
    recursive function
    :param data: the data from the call. can be a list or a dictionary, and this will try to extract from it
    :type data: dict,list
    :return: a __from_json(data) value, if it has 'cdl' inside the dictionary, or in the dictionaries in the list. 
             if it has no 'cdl' value, it returns a list of the words from the json
    :rtype: dict, list
    '''
    try:
        if isinstance(data, list):
            for pos in data:
                if "cdl" in data:
                    return __from_json(pos)
            return data
        elif isinstance(data, dict):
            return __from_json(data["cdl"])
    except KeyError:
        return []


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
        return []


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
        pass


def read_text(file: str):
    '''reads the file list, and count the number of lines and words

    :param file: the json file path
    :type file: str
    :return: counter of the words and the lines in the file
    :rtype: list
    '''
    counter = [0, 0, ""]

    try:
        for j in _get_data(file):
            if "label" in j:
                counter[0] += 1
                counter[2] = j["label"]
            else:
                counter[1] += 1
    except TypeError:
        pass
    finally:
        return counter


def get_catalog(lst: list, num: int = 5):
    if num == 0:
        return lst
    lst += glob.glob(f"jsons_unzipped/{'*/'*num}/X*.json", recursive=True)
    lst += glob.glob(f"jsons_unzipped/{'*/'*num}/P*.json", recursive=True)
    lst += glob.glob(f"jsons_unzipped/{'*/'*num}/Q*.json", recursive=True)
    return get_catalog(lst, num-1)


def get_langs(file: str):
    data = _get_data(file)
    data

    return (d["f"]["lang"] for d in data if d.get("f") and d["f"].get("lang")) if data else []


def append_langs(file: str):
    for l in get_langs(file):
        if l not in LANG_LIST:
            LANG_LIST.append(l)


def run_files(file: str):
    threading.Thread(None, append_langs, args=(file,)).start()


if __name__ == "__main__":
    d = get_catalog([], 7)
    for f in d:
        run_files(f)
    with open("langs_that_allowed.txt", "w", encoding="utf_8") as file:
        for i in LANG_LIST:
            file.write(f"{i}\n")
    print("no")
