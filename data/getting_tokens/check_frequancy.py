
from typing import Union
import pandas as pd
import numpy as np
import json
import glob
import os


LIST_OF_WORDS = []


def get_catalog(lst: list, num: int = 5):
    if num == 0:
        return lst
    lst += glob.glob(f"jsons_unzipped/{'*/'*num}/X*.json", recursive=True)
    lst += glob.glob(f"jsons_unzipped/{'*/'*num}/P*.json", recursive=True)
    lst += glob.glob(f"jsons_unzipped/{'*/'*num}/Q*.json", recursive=True)
    return get_catalog(lst, num-1)


def dict_places(lst: list):
    return {a.split('\\')[-1][:-5]: a for a in lst}


def _members(name: Union[str, list], recursive: bool = False):
    if isinstance(name, list):
        return _members(name[0], recursive)+_members(name[1:], recursive) if name else []
    if not recursive:
        return glob.glob(f"./jsons_unzipped/{name}/catalogu*.json")
    elif not glob.glob(f"./jsons_unzipped/{name}/catalogu*.json") and "corpusjson" not in glob.glob(f"./jsons_unzipped/{name}/")[0]:
        return _members(f"{name}/*", True)
    return _members(f"{name}/*")+glob.glob(f"./jsons_unzipped/{name}/catalogu*.json")


def _every_member(file: str):
    with open(file, 'r', encoding="utf_8") as f:
        members = json.load(f)["members"] if json.load(f).get("members") else {}
    return list(members.keys())
# def _read_member(file: str):


def get_texts(**ignore):
    projects = (pr for pr in os.listdir("jsons_unzipped") if not ignore.get("projects") or pr not in ignore["projects"])

    return projects


if __name__ == "__main__":
    # s = _members(get_texts(), True)
    # m = _members(get_texts(), True)
    # d = get_file(_every_member(m))
    s = dict_places(get_catalog([], 7))
    print(get_texts())
