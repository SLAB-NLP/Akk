from pathlib import Path

from tokenizers import BertWordPieceTokenizer
from tokenizers.pre_tokenizers import Whitespace

# from akkadian_bert.train_bert import ROBERTA_UNUSED_TOKENS_NUM  # TODO: delete line?
ROBERTA_UNUSED_TOKENS_NUM = 99


def train_tokenizer(model_dir: str, vocab_size: int, bert_input_file: str) -> None:
    """
    This function trains a tokenizer given a preprocessed input jsonl file.

    :param model_dir: The model's directory name
    :param vocab_size: The maximum size of the tokenizer's vocabulary
    :param bert_input_file: A path to write to the input bert file
    """

    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(lowercase=False)
    tokenizer.pre_tokenizer = Whitespace()

    # Customize training
    tokenizer.train(
        files=[bert_input_file],
        vocab_size=vocab_size,
        min_frequency=2,
    )
    Path(f"./{model_dir}/akk_tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(f"./{model_dir}/akk_tokenizer")


def change_unused_vocab_token(ind, new_token, tokenizer):
    del tokenizer.vocab[f'[unused{ind}]']
    tokenizer.vocab[new_token] = ind


def inject_new_tokens_to_tokenizer(new_tokens, tokenizer, special_tokens, max_tokens_to_add=ROBERTA_UNUSED_TOKENS_NUM):
    tokens_added = 0
    # copy_dict = tokenizer.vocab.copy()
    for token in new_tokens:
        if tokens_added >= max_tokens_to_add:
            break
        if token not in special_tokens and token not in tokenizer.vocab:
            tokens_added += 1
            # change_unused_vocab_token(tokens_added, token, tokenizer)
            tokenizer.vocab.pop(f'[unused{tokens_added}]', None)
            tokenizer.vocab[token] = tokens_added
    # tokenizer.vocab = copy_dict
