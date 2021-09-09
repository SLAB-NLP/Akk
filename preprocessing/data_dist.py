import glob
from collections import Counter
from typing import List

import sys
import os
import json

from preprocessing.scraping import JSONS_DIR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def split_data_by_properties(preprocessed_data_path: str, properties: List[str]):
    counters_dict = {prop: Counter() for prop in properties}

    all_members = get_all_catalogues_members()
    with open(preprocessed_data_path, 'r', encoding='utf-8') as f_data, open('extended_test_dataset.json', 'w',
                                                                             encoding='utf-8') as f_out:
        for data in f_data:
            cur_data_json = json.loads(data)
            if cur_data_json['id_text'] in all_members:
                cur_member_json = all_members[cur_data_json['id_text']]
                for prop in properties:
                    prop_val = cur_member_json.get(prop)
                    counters_dict[prop].update([prop_val])
                    cur_data_json[prop] = prop_val
            json.dump(cur_data_json, f_out)
            f_out.write('\n')
    print(counters_dict)
    return counters_dict


def get_all_catalogues_members():
    members = dict()
    all_catalogues_paths = glob.glob(f'{JSONS_DIR}/**/catalogue.json', recursive=True)
    for catalogue_path in all_catalogues_paths:
        with open(catalogue_path, 'r', encoding='utf-8') as json_file:
            try:
                cur_json = json.load(json_file)
            except Exception:
                continue
            members.update(cur_json.get('members', dict()))

    return members
