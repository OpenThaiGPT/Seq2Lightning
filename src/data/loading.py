import os
import datasets
import logging
import requests
import argparse
import dask.dataframe as dd

from multiprocessing.pool import ThreadPool
from functools import partial
from typing import List, Optional
from utils import convert_tsv_to_csv

logger = logging.getLogger(__name__)


def add_task_info(row):
    tasks = row["task"]
    langs = row["target_lang"]
    new_sources = [f'{task} to {lang}: {source}' for task, lang, source in zip(tasks, langs, row["source"])]
    return {"source": new_sources}


def split_dataset(dataset: datasets.Dataset,
                  val_frac: float=0.2, test_frac: float=0.1) -> datasets.DatasetDict:
    """Split dataset into train, val and test parts"""
    splitted = dataset.train_test_split(
        test_size=test_frac + val_frac,
        shuffle=True,
        seed=42)
    train_dataset, val_test_dataset = splitted['train'], splitted['test']
    splitted = val_test_dataset.train_test_split(
        test_size=test_frac
    )
    val_dataset, test_dataset = splitted['train'], splitted['test']
    dataset_dict = datasets.DatasetDict({
      'train': train_dataset,
      'test': test_dataset,
      'validation': val_dataset
     })
    return dataset_dict


def load_wikimatrix_gz(output_file: str, target_lang: str,
                       source_lang: str = "en") -> None:
    """Download a compressed TSV file containing parallel sentences from the WikiMatrix dataset"""
    url = f"https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.{source_lang}-{target_lang}.tsv.gz"
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, "wb") as file:
            file.write(response.content)
        logger.info(f"File downloaded successfully as '{output_file}'.")
    else:
        logger.info(f"Failed to download the file {output_file}")
        response.raise_for_status()


def load_wikimatrix_csv(dataset_dir: str, source_lang: str, target_languages: List[str],
                        max_rows: int, threshold: float,
                        remove_archive: bool = False) -> List[str]:
    """Download WikiMatrix parallel corpuses and save as csv files in dataset folder"""

    def load_wikimatrix_(lang, source_lang):
        nonlocal max_rows, threshold
        try:
            output_filename = f"wikimatrix_{source_lang}_{lang}"
            output_file_gz = os.path.join(dataset_dir, f"{output_filename}.tsv.gz")
            unzipped_filename = f"{output_filename}_{max_rows}.csv"
            if os.path.isfile(output_file_gz):
                print(f"Found file {output_file_gz}, no need to load again")
            else:
                load_wikimatrix_gz(output_file_gz, lang, source_lang)
            convert_tsv_to_csv(output_file_gz, unzipped_filename,
                               header_row=['margin_score', 'source', 'target'],
                               max_rows=max_rows, threshold=threshold)
            if remove_archive:
                os.remove(output_file_gz)
            return unzipped_filename
        except Exception as e:
            print(e)
            return None

    load_func = partial(load_wikimatrix_, source_lang=source_lang)
    with ThreadPool() as pool:
        dataset_files = pool.map_async(load_func, target_languages).get()
    dataset_files = [data_file for data_file in dataset_files if data_file is not None]
    return dataset_files


def load_wikimatrix_dataset(data_dir: str, source_lang: str,
                            target_languages: List[str],
                            max_rows: int, threshold: float,
                            remove_csv: bool = True) -> datasets.Dataset:
    """Download Wikimatrix for several languages and prepare for using in translation tuning"""

    def load_for_translation(csv_filename):
        dataframe = dd.read_csv(csv_filename)
        target_lang = csv_filename.split('_')[-2]  # output_file was like "wikimatrix_{source_lang}_{lang}_{maxrows}"
        dataframe['target_lang'] = target_lang
        dataframe = dataframe.persist()
        if remove_csv == True:
            os.remove(csv_filename)
        return dataframe

    files = load_wikimatrix_csv(data_dir, source_lang, target_languages,
                                max_rows=max_rows, threshold=threshold)
    with ThreadPool() as pool:
        dataframes = pool.map(load_for_translation, files)
    wiki_dataframe = dd.concat(dataframes).compute()
    wiki_dataset = datasets.Dataset.from_pandas(wiki_dataframe)
    wiki_dataset = wiki_dataset.add_column('task', ['translation'] * len(wiki_dataset))
    return wiki_dataset


def load_tedtalks(languages: List[str]) -> datasets.Dataset:
    """Download TedTalks dataset for translation"""

    def extract_source_target(row, source_lang, target_lang):
        return {'source': row['translation'][source_lang],
                'target': row['translation'][target_lang]}

    pairs = [(lang, 'en') for lang in languages if lang != 'en']
    all_parts = []
    for pair in pairs:
        part_dataset = datasets.load_dataset("ted_talks_iwslt", split='train',
                                             language_pair=pair, year='2014')
        extract_columns = partial(extract_source_target,
                                  source_lang=pair[0], target_lang=pair[1])
        part_dataset = part_dataset.map(extract_columns,
                                        remove_columns=['translation'])
        part_dataset = part_dataset.add_column('target_lang',
                                               [pair[1]] * len(part_dataset))
        all_parts.append(part_dataset)
    ted_dataset = datasets.concatenate_datasets(all_parts)
    ted_dataset = ted_dataset.add_column('task', ['translation'] * len(ted_dataset))
    return ted_dataset


def load_translation_dataset(data_dir: str,
                             source_lang: str, target_languages: List[str],
                             max_rows_per_table: int = 500000,
                             **params):
    os.makedirs(data_dir, exist_ok=True)
    wiki_threshold = params.get("wikimatrix_threshold", 1.04)
    wiki_dataset = load_wikimatrix_dataset(data_dir, source_lang, target_languages,
                                           max_rows=max_rows_per_table, threshold=wiki_threshold)
    ted_dataset = load_tedtalks(target_languages)
    translation_dataset = datasets.concatenate_datasets([wiki_dataset, ted_dataset])
    return translation_dataset


def load_mlsum(split: str, languages: list) -> datasets.Dataset:
    """Load mlsum dataset in several languages and prepare for summarization task"""
    all_parts = []
    existing_confs = datasets.get_dataset_config_names("mlsum")
    found_langs = list(set(existing_confs).intersection(languages))
    for lang in found_langs:
        dataset_part = datasets.load_dataset('mlsum', lang, split=split)
        dataset_part = dataset_part.add_column('target_lang', [lang] * len(dataset_part))
        dataset_part = dataset_part.add_column('task', ['summarization'] * len(dataset_part))
        dataset_part = dataset_part.rename_column('text', 'source')
        dataset_part = dataset_part.rename_column('summary', 'target')
        all_parts.append(dataset_part)

    concatenated_mlsum = datasets.concatenate_datasets(all_parts)
    return concatenated_mlsum


def load_cnn(split: str) -> datasets.Dataset:
    """Load cnn_dailymail dataset and prepare for summarization task"""
    cnn_dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', split=split)
    cnn_dataset = cnn_dataset.rename_column('article', 'source')
    cnn_dataset = cnn_dataset.rename_column('highlights', 'target')
    cnn_dataset = cnn_dataset.add_column('target_lang', ['en'] * len(cnn_dataset))
    cnn_dataset = cnn_dataset.add_column('task', ['summarization'] * len(cnn_dataset))
    return cnn_dataset


def load_summarization_dataset(split: str,
                                target_languages: List[str]) -> datasets.Dataset:
    """Load dataset for summarization task compiled from several sources"""
    mlsum_dataset = load_mlsum(split, target_languages)
    cnn_dataset = load_cnn(split)
    summ_dataset = datasets.concatenate_datasets([mlsum_dataset, cnn_dataset])
    return summ_dataset


def prepare_multilang_dataset(data_dir: str, target_languages: List[str],
              max_rows_per_table: int=40000,
              val_frac: float=0.2, test_frac: float=0.1,
              new_dataset_name: str = "multi_dataset") -> None:
    """Load, preprocess and save multilingual dataset with source language English"""
    source_lang = 'en'
    splits = ['train', 'validation', 'test']
    keep_columns = ['source', 'target', 'task', 'target_lang']
    data_translation = load_translation_dataset(data_dir, source_lang, target_languages, max_rows_per_table)
    data_translation = split_dataset(data_translation, val_frac, test_frac)
    for split in splits:
        part_summarization = load_summarization_dataset(split, target_languages)
        part_translation = data_translation[split]
        dataset_split = datasets.concatenate_datasets([part_summarization, part_translation]).shuffle()
        dataset_split = dataset_split.remove_columns(
            [col for col in dataset_split.column_names if col not in keep_columns]
        )
        dataset_split = dataset_split.map(add_task_info, batched=True)
        output_file = os.path.join(data_dir, f"{new_dataset_name}_{split}.parquet")
        dataset_split.to_parquet(output_file)


def run_dataset_loading():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Folder in which data will be saved")
    parser.add_argument("--target_languages", nargs="+", type=str, help="List of target languages abbreviations "
                                                                        "separated by spaces")
    parser.add_argument("--max_rows_per_table", type=int, default=40000, help="Maximum rows per language in wikimatrix")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Ratio of data to be allocated to a validation "
                                                                    "set during data splitting")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Ratio of data to be allocated to a test "
                                                                     "set during data splitting")
    parser.add_argument("--new_dataset_name", type=str, default="multi_nlp_dataset", help="Name for dataset files")
    args = parser.parse_args()
    prepare_multilang_dataset(data_dir=args.data_dir, target_languages=args.target_languages,
                 max_rows_per_table=args.max_rows_per_table, val_frac=args.val_frac,
                 test_frac=args.test_frac, new_dataset_name=args.new_dataset_name)


if __name__ == "__main__":
    run_dataset_loading()