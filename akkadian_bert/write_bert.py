import json
import random
from typing import Union

from akkadian_bert.utils import create_dirs_for_file


def write_bert_input_from_preprocessed(preprocessed_data_file: str, bert_input_file) -> None:
    """
    This function writes a bert input file from a given preprocessed jsonl data file.
    It assumes that every line in preprocessed_data_file is a json that contains a "bert_input" field which is a
    string of the text in a document.

    :param preprocessed_data_file: A path to a given preprocessed jsonl data file
    :param bert_input_file: A path to write to the input bert file
    """
    create_dirs_for_file(bert_input_file)
    with open(bert_input_file, "w", encoding='utf-8') as f_out:
        with open(preprocessed_data_file, "r", encoding='utf-8') as f_in:
            for line in f_in:
                f_out.write(json.loads(line)["bert_input"])
                f_out.write('\n')


def write_bert_files(preprocessed_akk_file: str, preprocessed_eng_file: Union[str, None], bert_akk_file: str,
                     bert_eng_file: str, bert_final_path: str, shorten: int = None):
    """
    This function writes the texts from an Akkadian and English preprocessed files in in text files used as an input
    for BERT-based models.

    :param preprocessed_akk_file: A preprocessed Akkadian jsonl data file
    :param preprocessed_eng_file: A preprocessed English jsonl data file
    :param bert_akk_file: A path to write the texts of the preprocessed Akkadian data file
    :param bert_eng_file: A path to write the texts of the preprocessed English data file
    :param bert_final_path: A path to write all the texts of the preprocessed files
    :param shorten: An integer representing the sample size to take from the Akkadian data
    """
    write_bert_input_from_preprocessed(preprocessed_akk_file, bert_akk_file)
    if preprocessed_eng_file:
        write_bert_input_from_preprocessed(preprocessed_eng_file, bert_eng_file)
    with open(bert_final_path, "w", encoding='utf-8') as f_out:
        with open(bert_akk_file, "r", encoding='utf-8') as f_akk:
            lines = f_akk.readlines()
            if shorten:
                lines = random.sample(lines, shorten)
            f_out.writelines(lines)
        if preprocessed_eng_file:
            with open(bert_eng_file, "r", encoding='utf-8') as f_eng:
                f_out.writelines(f_eng.readlines())
