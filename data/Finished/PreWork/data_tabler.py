import glob
import json
import numpy as np
import pandas as pd

INDEX = ['Name', 'display_name', 'genre', 'title', 'ruler', 'date', 'provenience', 'line_count', 'word_count',
         "Personal_Names", "aproximate_breaks", "number_of_dates", "number_of_eponyms", "places_of_dates_in_text", 'Relative_Path']
TEXTUS = pd.DataFrame(columns=INDEX, index=["Name"])

EXCEL_PATH = "data/Finished/PreWork/textdata.xlsx"

CSV_PATH = "data/Finished/PreWork/textdata.csv"


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
    FILE_DICT = {filus[filus.rfind("\\")+1:filus.find(".json")]: filus for filus in FILE_LIST}


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


def count_personal_names(word: dict) -> int:
    '''count the personal names in the file

    :param word: a dictionary that might be personal name
    :type word: dict
    :return: 0 if the word is not personal name, and 1 if it is
    :rtype: int
    '''
    try:
        return 1 if word.get("f") and word["f"].get("pos") and word["f"]["pos"] == "PN" else 0
    except TypeError:
        pass


def aproximate_breaks(word: dict) -> int:
    '''this function counts the APROXIMATE breaks in the texts. It does not concider the lenght of the break, size and shape, only if there is any

    :param word: a file to check
    :type file: str
    :return: count of the breaks
    :rtype: int
    '''

    try:
        return 1 if word.get("f") and word["f"].get("gdl") and word["f"]["gdl"][0].get("break") else 0
    except TypeError:
        pass


def __checkable_date_word(word: dict) -> bool:
    '''HELPER:__checkable_date_word checks if the word is checkable for get date

    :param word: a word to check if all the parameters in the dictionary are included
    :type word: dict
    :return: true if everything is fine for checking, false otherwise
    :rtype: bool
    '''
    return word.get("f") and word["f"].get("gw") and word["f"].get("pos")


def _detect_date(word: dict) -> int:
    '''HELPER: _detect_date detects if there is a date paricle in the text. if there is, it returns what date particle it is.

    :param word: a word dictionary to check
    :type word: dict
    :return: index of date particle, in the tuple (day, month, year), if it is a date, else, it returns false
    :rtype: int
    '''
    if __checkable_date_word(word):
        if word["f"]["gw"] == "day":
            return 0
        elif word["f"]["pos"] == "MN":
            return 1
        elif "eponym" in word['f']['gw']:
            return 2
    return False


def count_date_info(word: dict) -> np.array:
    '''count_date_info counts the dates in the text

    :param word: dict to check
    :type word: dict
    :return: the number of dates in the text in the array format of (day,month,year) as np.array
    :rtype: np.array
    '''
    counter_date = np.array([0, 0, 0])
    try:
        if word.get("f") and word["f"].get("gw") and word["f"].get("pos") and _detect_date(word):
            date_part = _detect_date(word)
            counter_date[date_part] += 1 if date_part else 0
        return counter_date
    except TypeError:
        pass


def place_of_dates(word: dict, line: str, last_line: str) -> str:
    '''place_of_dates checks where the dates appear in the text

    :param word: word to check if it is date
    :type word: dict
    :param line: line in the text
    :type line: str
    :param last_line: last line of the text
    :type last_line: str
    :return: place of the date in the line if it is not at the end, -1 if it is in the end
    :rtype: str
    '''
    '''place_of_dates 

    :param file: file to check
    :type file: str
    :return: list of places of dates in the text. if there is a date in the last line of the text, it will be converted to -1
    :rtype: list
    '''
    try:
        if word.get("f") and word["f"].get("gw") and word["f"].get("pos") and _detect_date(word):
            return line if line != last_line else -1
    except TypeError:
        pass


def count_eponyms(word: dict) -> int:
    '''count_eponyms counts the appearnce of the eponyms in the text, except in līmu lists, which are eponym lists

    :param word: word to check the number of eponyms
    :type word: dict
    :return: count of the eponyms mention in the text
    :rtype: int
    '''
    return 1 if _detect_date(word) == 2 else 0


def _ratio(var1: int, var2: int) -> float:
    '''ratio calculates the ratio between the two varibales

    :param var1: the first variable
    :type var1: int
    :param var2: secound variable
    :type var2: int
    :return: the ratio if var2 != 0, 0 otherwise
    :rtype: float
    '''
    return float(var1/var2) if var2 != 0 else 0


def main_data_loop(file: str) -> dict:
    '''main_data_loop is the main loop that runs over the file, and categorize and manipulatize it, for editing and entering to the data table

    :param file: path of the file for manipulize
    :type file: str
    :return: a dicitionary that contains all the data
    :rtype: dict
    '''
    text_counter = read_text(file)
    line_count = text_counter[0]
    word_count = text_counter[1]
    places_of_dates_in_text = []
    eponyms = 0
    number_of_dates = np.array([0, 0, 0])
    Aproximate_breaks = 0
    personal_names = 0
    last_line = text_counter[2]
    current_line = ""
    data = _get_data(file) if _get_data(file) else []
    for word in data:
        if word.get("label"):
            current_line = word["label"]
        personal_names += count_personal_names(word)
        Aproximate_breaks += aproximate_breaks(word)
        if current_line and place_of_dates(word, current_line, last_line):
            places_of_dates_in_text.append(place_of_dates(word, current_line, last_line))
        if _detect_date(word):
            number_of_dates += count_date_info(word)
        eponyms += count_eponyms(word)
    places_of_dates_in_text = places_of_dates_in_text if places_of_dates_in_text else None
    return {"line_count": line_count, "word_count": word_count, "places_of_dates_in_text": places_of_dates_in_text,
            "number_of_eponyms": eponyms, "number_of_dates": max(number_of_dates), "aproximate_breaks": Aproximate_breaks,
            "Personal_Names": personal_names, 'Word_ratio': _ratio(personal_names, word_count), 'Line_ratio': _ratio(personal_names, line_count),
            'Break_ratio': _ratio(personal_names, Aproximate_breaks), 'Fragmentary': _ratio(word_count, Aproximate_breaks)}


def dict_a_file(Key: str):
    '''the function gets an entry from the user or the function: a key from the files. 
    the function adds the data to a temporal dict to fulfill the entry with any metadata and calculations of the words, lines,
    personal names, and the ratio between the PN and the lines, and the PN and the words. finally the function will add the relative path
    of the file to the dictionary, and will call to add the temporal_dictionary to the DataFrame TEXTUS.

    if the key is not in the catalogue, it will skip the metadata import

    :param Key: the name of the text
    :type Key: str
    '''
    global METADATA, FILE_DICT, INDEX
    data_dict = METADATA[Key] if METADATA.get(Key) and METADATA and FILE_DICT else None
    temporal_dictionary = {ind: data_dict[ind] for ind in INDEX if ind in data_dict} if data_dict is not None else {}
    temporal_dictionary.update(main_data_loop(FILE_DICT[Key]))
    temporal_dictionary.update({"Name": Key})
    temporal_dictionary['Relative_Path'] = FILE_DICT[Key].split("ancient-text-processing")[-1][1:].replace(
        "\\", "/") if "ancient-text-processing" in FILE_DICT[Key] else FILE_DICT[Key].replace("\\", "/")
    append_textus(temporal_dictionary)


def dict_file():
    '''This function creates a loop that going throu all of the keys of the FILE_DICT, and calls "dict_a_file" function
    '''
    for filename in FILE_DICT:
        dict_a_file(filename)


def dict_a_metadata(data: str):
    '''the function gets an entry from the user or the function: a key from the metadata, which is NOT in the files.
    the function adds the data to fulfill the entry with the metadata. finally the function adds the key name to the temporal_dictionary
    and call the append_textus function to add it to the TEXTUS DataFrame

    :param data: a key that in the METADATA dictionary
    :type data: str
    '''
    global METADATA, INDEX, FILE_DICT
    if data not in FILE_DICT:
        metadata = METADATA[data]
        temporal_dictionary = {ind: metadata[ind] for ind in INDEX if ind in metadata}
        temporal_dictionary["Name"] = data
        append_textus(temporal_dictionary)


def dict_metadata():
    '''this function creates a loop that running throu all the keys of the METADATA dicionary and calls "dict_a_metadata" function
    '''
    global METADATA, INDEX, FILE_DICT
    for data in METADATA:
        dict_a_metadata(data)


def append_textus(dict_from_file: dict):
    '''this function gets a dictionary, and appends it to the TEXTUS DataFrame

    :param dict_from_file: a dictionary of the desired values to add to the DataFrame
    :type dict_from_file: dict
    '''
    global TEXTUS
    TEXTUS = TEXTUS.append(dict_from_file, True)


if __name__ == "__main__":
    set_project()
    dict_file()
    dict_metadata()
    path = TEXTUS['Relative_Path']
    TEXTUS.drop('Relative_Path', 1, inplace=True)
    TEXTUS.insert(TEXTUS.columns.size, 'Relative_Path', path)
    with pd.ExcelWriter(EXCEL_PATH) as excel:
        TEXTUS.loc[:,
                   ['Name', 'display_name', 'genre', 'title', 'ruler', 'date', 'provenience', 'Fragmentary',
                    'line_count', 'word_count', 'Personal_Names', 'aproximate_breaks', 'Line_ratio', 'Word_ratio',
                    'Break_ratio', "number_of_dates", "number_of_eponyms", "places_of_dates_in_text",
                    'Relative_Path']].to_excel(excel, sheet_name="Base", index=False)
    TEXTUS.to_csv(CSV_PATH, index=False)
