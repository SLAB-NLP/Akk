import pandas as pd

# with open("lemma.txt","r",encoding="utf_8") as f:
#     lemma_preprocessing = {l.strip("'").strip("'\n"):{} for l in f}

lemma_preprocessing = {}


def _read_text(file: str):
    """HELPER: extracts the list of the text from the file
    
    :param file: file path of the json file
    :type file: str
    :return: list of the words, lines and other that in the json
    :rtype: list
    """
    with open(file, "r", encoding="utf_8") as File:
        try:
            return eval(File.read())["cdl"][0]["cdl"][-1]["cdl"][0]["cdl"]
        except KeyError:
            pass
        except SyntaxError:
            pass


def _get_ref(word: dict):
    """HELPER: extracts the reference from the dictionary

    :param word: the node in the list of the text
    :type word: dict
    :return: the reference and the exact part of the file
    :rtype: str
    """
    try:
        if "ref" in word:
            return word["ref"]
    except TypeError:
        pass


def _get_sig_data(word: dict):
    """HELPER: extracts if the sig is in word dictionary

    :param word: dictionary from the json word
    :type word: dict
    :return: if sig, ':' and '$' in the word dictionary, it returns it. otherwise it returns False.
    :rtype: list
    """
    if "sig" in word:
        if ":" in word["sig"] and "$" in word["sig"]:
            return [word["sig"][word["sig"].find(":") + 1:word["sig"].find("=")],
                    word["sig"][word["sig"].find("$") + 1:]]
    return False


def _get_writing(word: dict):
    """HELPER: extracts the writing from the word

    :param word: dictionary from the json word
    :type word: dict
    :return: the writing of the word
    :rtype: str
    """
    try:
        if _get_sig_data(word):
            return _get_sig_data(word)[0]
        return None
    except TypeError:
        pass


def _get_linked_transliteration(word: dict):
    """HELPER: extracts the linked transliteration from the word

    :param word: dictionary from the json word
    :type word: dict
    :return: the linked transliteration of the word
    :rtype: str
    """

    try:
        if _get_sig_data(word):
            return _get_sig_data(word)[1]
        return None
    except TypeError:
        pass


def _get_from_f(word: dict, key: str):
    """HELPER: extracts the key from the word dictionary

    :param word: dictionary from the json word
    :type word: dict
    :param key: key that needed to be found in the f dictionary that in word
    :type key: str
    :return: value of key in the f key in word dictionary
    :rtype: str
    """
    try:
        if "f" in word:
            if key in word["f"]:
                return word["f"][key]
    except TypeError:
        pass


def _get_cf(word: dict):
    """HELPER: extracts the cf from the word dictionary

    :param word: dictionary from the json word
    :type word: dict
    :return: dictionary of the cf as key, and value of cf in the f key in word dictionary
    :rtype: str
    """
    try:
        return _get_from_f(word, "cf")
    except TypeError:
        pass


# def __pure_cf(word: dict):
#     """HELPER: extracts the cf only

#     Args:
#         word (dict): dictionary of the word

#     Returns:
#         the word itself
#     """
#     try:
#         if "cf" in _get_cf(word):
#             return _get_cf(word)["cf"]
#     except TypeError:
#         pass

def _get_pos(word: dict):
    """HELPER: extracts the pos from the word dictionary

    :param word: dictionary from the json word
    :type word: dict
    :return: dictionary of the pos as key, and value of pos in the f key in word dictionary
    :rtype: str
    """
    try:
        return _get_from_f(word, "pos")
    except TypeError:
        pass


def _get_norm(word: dict):
    """HELPER: extracts the norm from the word dictionary

    :param word: dictionary from the json word
    :type word: dict
    :return: dictionary of the norm as key, and value of norm in the f key in word dictionary
    :rtype: str
    """
    try:
        return _get_from_f(word, "norm")
    except TypeError:
        pass


def _checker(word: dict):
    """checks if the 'word' dictionary is fine

    :param word: the node in the list of the text
    :type word: dict
    :return: if "f", "ref" and "sig" in word, returns true, else, returns false
    :rtype: bool
    """
    if "f" in word and "ref" in word and "sig" in word:
        return True
    return False


def analyze_file(file: str):
    """analyzes the file, to extract from it all the "ref"s, "transliterate"s, "writing"s, "pos"es, and "norm"s; put everything in the relevant list

    :param file: file path
    :type file: str
    """
    global lemma_preprocessing
    for i in _read_text(file):
        if _checker(i):
            if _get_cf(i) not in lemma_preprocessing:
                lemma_preprocessing[_get_cf(i)] = []
            lemma_preprocessing[_get_cf(i)].append(
                {"ref": _get_ref(i), "transliterate": _get_linked_transliteration(i), "writing": _get_writing(i),
                 "pos": _get_pos(i), "norm": _get_norm(i)})
            # lemma_preprocessing[_get_cf(i)][_get_ref(i)]["transliterate"] = _get_linked_transliteration(i)
            # lemma_preprocessing[_get_cf(i)][_get_ref(i)]["writing"] = 
            # lemma_preprocessing[_get_cf(i)][_get_ref(i)]["pos"] = _get_pos(i)
            # lemma_preprocessing[_get_cf(i)][_get_ref(i)]["norm"] = _get_norm(i)


def files_to_analyze(file_list: str):
    """a loop that analyzes the list of paths 

    :param file_list: a file path that stores all the file pathes
    :type file_list: str
    """
    with open(file_list, 'r', encoding='utf_8') as f_i:
        for i in f_i:
            try:
                analyze_file(i.strip("\n"))
            except KeyError:
                continue
            except TypeError:
                continue


if __name__ == "__main__":
    files_to_analyze(r"D:\Drive\לימודים\מאגרי מידע\זמני\ancient-text-processing\paths of the jsons.txt")
    with open('../anala.txt', 'w', encoding='utf_8') as f_o:
        f_o.write(str(lemma_preprocessing))
    with open('lemma_len.txt', 'w', encoding='utf_8') as file:
        for i in lemma_preprocessing:
            file.write(f"\n{i}: {len(lemma_preprocessing[i])}")
    # with open('analysis of the lemmas, every reference.txt','r',encoding='utf_8') as file:
    #     s_file = eval(file.read())
    #     with open('stat.csv','w',encoding='utf_8') as f:
    #         f.write("word, count\n")
    #         for i in s_file :
    #             f.write(f"{i},{len(s_file[i])}\n")
