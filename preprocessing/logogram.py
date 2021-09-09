import glob
import json


def set_project(project_list: list = ["rinap", "saao"], *project: str):
    global METADATA, FILE_LIST, FILE_DICT
    if project_list is None:
        for name in project:
            for gl in [*glob.glob(f"jsons_unzipped/{name}/catal*.json"),
                       *glob.glob(f"jsons_unzipped/{name}/*/catal*.json")]:
                with open(gl, encoding="utf_8") as file:
                    METADATA.update(json.load(file)["members"])
                    FILE_LIST.append(glob.glob(f"jsons_unzipped/{name}/*/corpusjson/*.json", recursive=True))
    elif len(project_list):
        for name in project_list:
            for gl in [*glob.glob(f"jsons_unzipped/{name}/catal*.json"),
                       *glob.glob(f"jsons_unzipped/{name}/*/catal*.json")]:
                with open(gl, encoding="utf_8") as file:
                    METADATA.update(json.load(file)["members"])
                    FILE_LIST += glob.glob(f"jsons_unzipped/{name}/*/corpusjson/*.json", recursive=True)
    else:
        project_list += list(project)
        for name in project_list:
            for gl in [*glob.glob(f"jsons_unzipped/{name}/catal*.json"),
                       *glob.glob(f"jsons_unzipped/{name}/*/catal*.json")]:
                with open(gl, encoding="utf_8") as file:
                    METADATA.update(json.load(file)["members"])

            FILE_LIST += glob.glob(f"jsons_unzipped/{name}/*/corpusjson/*.json", recursive=True)
    FILE_DICT = {filus[filus.rfind("\\") + 1:filus.find(".json")]: filus for filus in FILE_LIST}


METADATA = {}
FILE_LIST = []
FILE_DICT = {}


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
