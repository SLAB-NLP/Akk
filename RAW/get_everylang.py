import pandas as pd
import json
import glob
import multiprocessing as mp
import threading

def get_files():





def get_catalog(lst: list, num: int = 5):
    if num == 0:
        return lst
    lst += glob.glob(f"jsons_unzipped/{'*/'*num}/cata*.json", recursive=True)
    return get_catalog(lst, num-1)


def get_file(lst: list):
    filedict = {}
    for l in lst:
        try:
            with open(l, "r", encoding="utf_8") as file:
                data = json.load(file)["members"]
            filedict.update({d: f'jsons_unzipped/{data[d]["project"]}/corpusjson/{d}.json' for d in data})
        except IndexError:
            continue
        except KeyError:
            continue
    return filedict


cats = get_catalog([], 7)
cols = [c.split('\\')[-2] for c in cats]
periods = []
langs = []
files_and_paths = get_file(cats)
files_to_project = {file:file.split('/')[-3] for file in files_and_paths}

Files_to_project = mp.Manager().dict(files_to_project)

# def read_and_check(dct: dict):


def get_period(elem: dict):
    return f'{elem["period"]}' if elem.get("period") else ""


def get_language(elem: dict):
    return f'{elem["language"]}' if elem.get("language") else ""


def files_write(langs: list, period: list):
    with open("data/lists/periob ds.txt", "w", encoding="utf_8") as file:
        file.writelines(period)
    with open("data/lists/langhjs.txt", "w", encoding="utf_8") as file:
        file.writelines(langs)


def create_lists():
    global langs, periods, cats, cols, files_and_paths
    for c in cats:
        try:
            with open(c, "r", encoding="utf_8") as file:
                data = json.load(file)["members"]
            for d in data:
                periods.append(get_period(data[d]))
                langs.append(get_language(data[d]))
        except:
            continue
    langs = list(set(langs))
    periods = list(set(periods))
