from typing import List, Dict
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from preprocessing.main_preprocess import SUPERSCRIPTS_TO_UNICODE_CHARS, remove_superscripts

CLASSIFIERS = {"{uru}": "PLACE",
               "{iri}": "PLACE",
               "{eri}": "PLACE",
               "{ki}": "PLACE",
               "{kur}": "PLACE",
               "{f}": "NAME",
               "{1}": "NAME",
               "{m}": "NAME",
               "{d}": "GOD"}

PLACE_CLASSIFIERS = [place for place in CLASSIFIERS if CLASSIFIERS[place] == "PLACE"]

NAME_CLASSIFIERS = [name for name in CLASSIFIERS if CLASSIFIERS[name] == "NAME"]


def get_list_pse_words() -> dict:
    """
    Creates a dictionary of lists based pseudo-words that exists in the data

    :list_pse_words: the dictionary that will be returned
    :lists: a list of the file names
    :pse_words: list of the pseudo-words
    :return: a dict of the pseudo-word
    :rtype: dict
    """
    pass
    # list_pse_words = dict()
    # lists = ["Names", "Divine Names", "Places"]
    # pse_words = ["NAME", "GOD", "PLACE"]
    # for lst, pse_word in zip(lists, pse_words):
    #     with open(f'data/lists/{lst}.txt', 'r', encoding='utf_8') as file:
    #         names = eval(file.read())
    #         for name in names:
    #             list_pse_words.update({n: pse_word for n in names[name]})
    # return list_pse_words


pse_words_dict: Dict = get_list_pse_words()


def get_index_classifier(word: str) -> List:
    """
    HELPER: this function finds the classifiers in a given word.

    :param word: a word to check, LOWER CASE ONLY!!!
    :type word: str
    :return: a list of the classifiers in the word, sorted by the order in which they appear in the word.
    :rtype: list
    """
    classifiers_in_word = [(word.find(classifier), classifier) for classifier in CLASSIFIERS if classifier in word]
    return [classifier for (_, classifier) in sorted(classifiers_in_word)]


def is_place(word: str, word_classifiers) -> bool:
    """
    HELPER: checks if a given word from ORACC is a place.
    If a word ends with "{ki}" or its first classifier is a place classifier, then it is considered as a place word.

    :param word: a word to check, LOWER CASE ONLY!!!
    :type word: str
    :param word_classifiers: a list of the classifiers in the word, sorted by the order in which they appear in the word
    :type word_classifiers: list
    :return: True iff the word is considered a place.
    :rtype: bool
    """
    return word.endswith("{ki}") or word_classifiers[0] in PLACE_CLASSIFIERS


def is_personal_name(word: str, word_classifiers) -> bool:
    """
    HELPER: checks if a given word from ORACC is a personal name.
    A word is considered a personal name of it contains a name classifier or it starts with "{d}" and has multiple '-'.

    :param word_classifiers: a list of the classifiers in the word, sorted by the order in which they appear in the word
    :param word: a word to check, LOWER CASE ONLY!!!
    :type word: str
    :return: NAME if it is a name, GOD if it is a god, otherwise, the word itself.
    :rtype: bool
    """

    starts_with_name_classifier = word.startswith(word_classifiers[0]) and word_classifiers[0] in NAME_CLASSIFIERS
    return starts_with_name_classifier or (word.startswith("{d}") and word.count('-') > 1)


def is_god_name(word: str):
    """
    This function checks if a given word from ORACC is a name of a god.
    A word is considered a god name if it starts with "{d}" and it contains at most one '-'.
    
    :param word: A given word
    :type word: str
    :return: True iff the given word is a god name.
    """
    return word.startswith("{d}") and word.count('-') <= 1


def convert_to_pseudo_word(word: str) -> str:
    """
    Main function: returns the pseudo-word for a given word (if there is such).
    It assumes that classifiers are always inside curly brackets, as in ORACC's implementation.

    :param word: a given word to check
    :type word: str
    :return: PLACE if it is a place, NAME if it is a name, GOD if it is a god, the word itself otherwise.
    :rtype: str
    """
    if "{" in word and "}" in word:
        word_lower_case = word.lower()
        classifiers_in_word = get_index_classifier(word_lower_case)
        if classifiers_in_word:
            if is_place(word_lower_case, classifiers_in_word):
                return "PLACE"
            if is_personal_name(word_lower_case, classifiers_in_word):
                return "NAME"
            if is_god_name(word_lower_case):
                return "GOD"
    return word


def convert_to_listbase_pseudo_word(word: str) -> str:
    """
    Returns the pseudo-word for the word, if it exists in the lists.

    :param word: a given word to check
    :type word: str
    :return: PLACE if it is a place, NAME if it is a name, GOD if it is a god, the word itself otherwise.
    :rtype: str
    """
    return pse_words_dict.get(word, default=word)


def decide_pse_word(word: str, list_based: bool = False) -> str:
    """
    Main Function: return the pseudo-word, and decide between list based pseudo-words and determinative-structural
    pseudo-word

    :param word: a given word to check
    :type word: str
    :param list_based: to base on list, or not, defaults to False
    :type list_based: bool, optional
    :return: PLACE if it is a place, NAME if it is a name, GOD if it is a god, the word itself otherwise.
    :rtype: str
    """
    determin_pse_word = convert_to_pseudo_word(word)
    if list_based:
        list_based_pse_word = convert_to_listbase_pseudo_word(word)
        if determin_pse_word == list_based_pse_word or (list_based_pse_word == word and determin_pse_word != word):
            return determin_pse_word
        return list_based_pse_word
    return determin_pse_word
