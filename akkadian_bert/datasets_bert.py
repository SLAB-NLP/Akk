import os

import numpy as np
import torch


class ORACCDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, block_size, missing_sign_encoding: int,
                 encode_only_first_token_in_word: bool = False,
                 ignore_missing=True):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        with open(file_path, 'r', encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.batch_encodings = tokenizer(lines, add_special_tokens=True, padding=True, truncation=True, max_length=block_size)
        self.labels = get_enc_labels(
            encodings=self.batch_encodings,
            missing_sign_encoding=missing_sign_encoding,
            encode_only_first_token_in_word=encode_only_first_token_in_word,
            ignore_missing=ignore_missing
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.batch_encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_enc_labels(encodings, missing_sign_encoding: int, encode_only_first_token_in_word: bool = False,
                   ignore_missing=True):
    encoded_labels = []

    for labels in encodings.encodings:
        # create an empty array of -100 (-100 is ignored during training)
        enc_labels = np.ones(len(labels.offsets), dtype=int) * -100
        arr_offset, arr_labels = np.array(labels.offsets), np.array(labels.ids)

        offset_indices = np.ones(len(labels.offsets), dtype=bool)
        if ignore_missing:
            offset_indices &= arr_labels != missing_sign_encoding

        if encode_only_first_token_in_word:
            # set labels whose first offset position is 0 and the second is not 0
            offset_indices &= (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)

        enc_labels[offset_indices] = arr_labels[offset_indices]
        encoded_labels.append(enc_labels.tolist())

    return encoded_labels
