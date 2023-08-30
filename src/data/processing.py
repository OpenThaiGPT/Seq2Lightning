import re
import os
import glob
import argparse
import logging
import unicodedata
import transformers
from typing import List
import dask.dataframe as dd
from collections.abc import Callable
from utils import check_language


logger = logging.getLogger(__name__)


def extract_tokens(input_string: str) -> List[str]:
    """Extract words and puntuation from a sentence with regex"""
    pattern = r'\w+|[.,!?;]'
    result = re.findall(pattern, input_string)
    return result


def extract_tokens_tokenizer(input_string: str,
                             tokenizer: transformers.PreTrainedTokenizerBase) -> List[int]:
    """Extract tokens from a sentence with transformers tokenizer"""
    return tokenizer.encode(input_string, add_special_tokens=False, padding=False)


def accept_length(sent_1: str, sent_2: str, max_sent_length: int,
                  tokenize_func: Callable, ratio: float = 2.) -> bool:
    """Check if two sentences tokens lengths differ more than given ratio and aren't too long"""
    if len(sent_1) > max_sent_length or len(sent_2) > max_sent_length:
        return False
    length_1 = len(tokenize_func(sent_1))
    length_2 = len(tokenize_func(sent_2))
    if length_1 > (length_2 * ratio) or length_2 > (length_1 * ratio):
        return False
    return True


def check_symbols(input_string: str, nonletter_ratio: float = 0.5) -> bool:
    """Check if symbols and words in input string look acceptable"""
    letters = [symb for symb in input_string.lower()
               if unicodedata.category(symb) == 'Ll']
    if len(letters) < (len(input_string) * nonletter_ratio):  # more non-letter symbols than the specified ratio
        return False
    if len(re.findall("[а-я|a-z]{1,}(?:[-|'|&]?[а-я|a-z]){40,}", input_string.lower())) > 0:
        return False
    if len(re.findall('(?:\d+[ |.]\d+){5,}', input_string)) > 0:
        return False
    return True


def check_row_acceptance(input_row, max_sent_length: int,
                         max_text_length: int, max_summary_length: int) -> bool:
    valid = True
    if not check_symbols(input_row['target']) or not check_symbols(input_row['source']):
        return False
    if not check_language(input_row['target'],
                          assumed_lang=input_row['target_lang']):
        return False
    if input_row['task'] == 'translation':
        valid = accept_length(input_row['target'], input_row['source'],
                              tokenize_func=extract_tokens, max_sent_length=max_sent_length)
    elif len(input_row['target']) > max_summary_length or len(input_row['source']) > max_text_length:
        valid = False
    return valid


def filter_nlp_dataset(dataset: dd.DataFrame,
                       max_sent_length: int, max_text_length: int,
                       max_summary_length: int) -> dd.DataFrame:
    accept = dataset.apply(check_row_acceptance, max_sent_length=max_sent_length,
                           max_text_length=max_text_length,
                           max_summary_length=max_summary_length,
                           axis=1, meta=('result', bool))
    filtered = dataset.loc[accept]
    return filtered


def preprocess_dataset(dataset_dir: str, output_dir: str,
                       max_sent_length: int, max_text_length: int,
                       max_summary_length: int) -> None:
    """Filter and clean dataset and save as parquet"""
    files_pattern = os.path.join(dataset_dir, "*.parquet")
    data_files = glob.glob(files_pattern)
    if not data_files:
        logger.error("Dataset files not found")
        return
    data_output_dir = output_dir if output_dir is not None else dataset_dir
    for data_file in data_files:
        data_filename = os.path.basename(data_file)
        output_file = os.path.join(data_output_dir, data_filename)
        dataset = dd.read_parquet(data_file)
        filtered = filter_nlp_dataset(dataset, max_sent_length, max_text_length,
                                      max_summary_length).compute()
        filtered.to_parquet(output_file, write_index=False)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Folder with data to be processed")
    parser.add_argument("--output_dir", type=str, help="Folder in which processed dataset will be stored")
    parser.add_argument("--max_sent_length", type=int, default=150, help="Maximum sentence length (in tokens)")
    parser.add_argument("--max_text_length", type=int, default=400, help="Maximum source text length (in tokens)")
    parser.add_argument("--max_summary_length", type=int, default=250, help="Maximum summary length (in tokens)")
    args = parser.parse_args()
    preprocess_dataset(dataset_dir=args.dataset_dir, output_dir=args.output_dir,
                       max_sent_length=args.max_sent_length, max_text_length=args.max_text_length,
                       max_summary_length=args.max_summary_length)


if __name__ == "__main__":
    run()
