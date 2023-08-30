import os
import torch
import csv
import glob
import logging
import argparse
import numpy as np
import dask.dataframe as dd
from typing import List, Any, Optional, Tuple
from tqdm.auto import tqdm, trange
from collections import Counter
from functools import partial
from transformers import (
    PreTrainedTokenizer, PreTrainedModel,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration, T5Tokenizer
)
from multiprocessing import Pool
from src.utils import run_shell_cmd, untar_archive, load_yaml_config

logger = logging.getLogger(__name__)

try:
    config_path = os.environ["config_path"]
except KeyError:
    script_directory = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_directory, '../configs/project_confs.yaml')
configs = load_yaml_config(config_path)
PROJECT_LANGS = configs['PROJECT_LANGS']
LANG_ARCHIVES = configs['LANG_ARCHIVES']


def load_spmp_proto() -> None:
    """Download a Protocol schema file for the SentencePiece and compile into Python code """
    run_shell_cmd("wget https://raw.githubusercontent.com/google/sentencepiece/master/src/sentencepiece_model.proto",
                  verbose=False)
    run_shell_cmd("protoc --python_out=. sentencepiece_model.proto")
    run_shell_cmd("rm sentencepiece_model.proto")


def find_corpus_file(tar_output_dir: str) -> str:
    """Find file with sentences from Leipzig corpus archive"""
    if not os.path.isdir(tar_output_dir):
        logging.error(f"Tar output dir not found: {tar_output_dir}")
        raise ValueError("Tar output dir nor existed")
    files_dir = os.listdir(tar_output_dir)[0]
    corpus_path = os.path.join(tar_output_dir, files_dir)
    results = glob.glob('*sentences.txt', root_dir=corpus_path)
    if len(results) > 0:
        return os.path.join(corpus_path, results[0])
    else:
        raise ValueError("Necessary file wasn't found")


def prepare_leipzig_corpus(lang: str, working_dir: str) -> str:
    """Unarchive leipzig corpus and return path to texts file"""
    output_dir = os.path.join(working_dir, f'{lang}_leipzig_corpus')
    untar_archive(LANG_ARCHIVES[lang], output_dir)
    data_filename = find_corpus_file(output_dir)
    return data_filename


def build_tokens_counter(dt: dd.DataFrame, tokenizer: PreTrainedTokenizer,
                         batch_size: int = 16) -> Counter:
    """Build collections.Counter for token frequencies"""
    counter = Counter()
    for i in tqdm(range(0, len(dt), batch_size)):
        batch = dt.text[i:i + batch_size].to_list()
        encodings = tokenizer.batch_encode_plus(batch, padding="longest", return_tensors="np")
        flat_input_ids = np.array(encodings["input_ids"]).flatten()
        counter.update(flat_input_ids)
    return counter


def common_tokens_leipzig(working_dir: str,
                          project_langs: List[str], tokenizer: PreTrainedTokenizer,
                          tokens_per_lang: int = 50000) -> List[int]:
    """Find most frequent tokens for each language from leipzig corpuses"""
    prepare_one_corpus = partial(prepare_leipzig_corpus, working_dir=working_dir)
    with Pool() as pool:
        text_filenames = pool.map(prepare_one_corpus, PROJECT_LANGS)
    new_tokens = set()
    for filename in text_filenames:
        dt = dd.read_csv(filename, sep='\t', header=None,
                         quoting=csv.QUOTE_NONE, names=['idx', 'text']).compute()
        lang_counter = build_tokens_counter(dt, tokenizer)
        common_tokens = [token for token, _ in lang_counter.most_common(tokens_per_lang)]
        new_tokens.update(common_tokens)
    new_tokens.update([i for i in range(0, 256 + 3)])  # we want to keep byte-level symbols
    return sorted(list(new_tokens))


def cut_model_for_lm(model: PreTrainedModel,
                     tokenizer: PreTrainedTokenizer,
                     new_tokens: List[int],
                     new_model_name: str,
                     ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Cut Seq2Seq model vocabulary leaving only selected tokens"""
    if not hasattr(model, 'lm_head'):
        raise ValueError("Passed model doesn't have lm_head")

    load_spmp_proto()
    import sentencepiece_model_pb2 as spmp

    new_size = len(new_tokens)
    new_emb = torch.nn.Embedding(new_size, model.shared.embedding_dim)
    new_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=new_size, bias=False)
    for new_id, old_id in enumerate(new_tokens):
        new_emb.weight.data[new_id] = model.shared.weight.data[old_id]
        new_head.weight.data[new_id] = model.lm_head.weight.data[old_id]
    model.shared.weight = new_emb.weight
    model.lm_head.weight = new_head.weight
    model.config.__dict__['vocab_size'] = new_size
    model.config.__dict__['_name_or_path'] = new_model_name

    smp = tokenizer.sp_model.serialized_model_proto()
    proto = spmp.ModelProto()
    proto.ParseFromString(smp)
    new_pieces = [proto.pieces[idx] for idx in new_tokens]
    for i, p in enumerate(new_pieces):
        proto.pieces[i].piece = p.piece
        proto.pieces[i].score = p.score
        proto.pieces[i].type = p.type
    n = len(new_pieces)
    for i in trange(len(proto.pieces) - n):
        proto.pieces.pop(len(proto.pieces) - 1)
    return model, proto


def cut_t5_based_model(model_name_or_path: str,
                 working_dir: str,
                 tokens_per_lang: int,
                 new_model_name: Optional[str] = None,
                 save_path: Optional[str] = None):
    """
    Cut vocabulary of T5-based model (including MT5), saving only most common tokens for each language

    :param model_name_or_path: Model name from huggingface hub or local path
    :param working_dir: Folder where to store temporary files
    :param tokens_per_lang: Maximum number of most frequent tokens to retain per language (default is 40000)
    :param new_model_name: Name for a new model (optional)
    :param save_path: Local path to save new model and tokenizer (optional)
    """
    if not os.path.isdir(working_dir):
        raise ValueError(f"{working_dir} doesn't exist")
    if new_model_name is None:
        new_model_name = model_name_or_path
    project_langs = PROJECT_LANGS
    tokenizer_path = os.path.join(working_dir, 'new_sp.model')
    if "mt5" in model_name_or_path:
        model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    md_max_length = model.config.d_model
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path,
                                             model_max_length=md_max_length, legacy=False)
    new_tokens = common_tokens_leipzig(working_dir, project_langs,
                                       tokenizer, tokens_per_lang=tokens_per_lang)
    model, tokenizer_proto = cut_model_for_lm(model, tokenizer, new_tokens,
                                              new_model_name=new_model_name)
    with open(tokenizer_path, 'wb') as f:
        f.write(tokenizer_proto.SerializeToString())
    new_tokenizer = T5Tokenizer(tokenizer_path, extra_ids=0, legacy=False)
    logger.info(f"New vocab length: {len(new_tokenizer)}")
    if save_path is not None and os.path.isdir(save_path):
        new_tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
    return model, new_tokenizer


def run_mt5_cropping():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Model name from huggingface hub or local path")
    parser.add_argument("--working_dir", type=str, help="Folder where to store temporary files")
    parser.add_argument("--tokens_per_lang", type=int, default=40000,
                        help="Maximum number of most frequent tokens to retain per language")
    parser.add_argument("--new_model_name", type=str, default=None, help="Name for a new model (optional)")
    parser.add_argument("--save_path", type=str, default=None, help="Local path to save new model and tokenizer")
    args = parser.parse_args()
    cut_t5_based_model(mt5_name_or_path=args.model_name_or_path,
                       working_dir=args.working_dir,
                       tokens_per_lang=args.tokens_per_lang,
                       new_model_name=args.new_model_name,
                       save_path=args.save_path)


if __name__ == "__main__":
    run_mt5_cropping()
