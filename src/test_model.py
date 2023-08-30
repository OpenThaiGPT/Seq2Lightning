import argparse
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_sequences(model: AutoModelForSeq2SeqLM,
                       tokenizer: AutoTokenizer,
                       text: np.ndarray, **gen_params) -> List[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer.batch_encode_plus(text.tolist(), padding="longest",
                                       max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(inputs["input_ids"].to(device), **gen_params)
    generated_texts = tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
    return generated_texts


def run_test_on_csv(model_path: str,
                    input_csv: str, input_column: str,
                    output_csv: Optional[str] = None,
                    batch_size: Optional[int] = 4,
                    **generation_params) -> None:
    """
    Test transformers Seq2Seq model generation on texts from csv file
    :param model_path: Path to pretrained model
    :param input_csv: Path to csv with texts
    :param input_column: Name of the column with input texts (default is "input_text")
    :param output_csv: Path to csv to save generated texts, if not provided, they will be saved in the original csv (optional)
    :param batch_size: Size of generation batches (default is 4)
    :param generation_params: Parameter for generate() method
    :return None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model.config.d_model)
    data = pd.read_csv(input_csv)
    if output_csv is None:
        output_csv = input_csv
    rows = data.input_text.values
    generated = []
    for batch_start in range(0, len(rows), batch_size):
        batch_texts = rows[batch_start:(batch_start + batch_size)]
        generated.extend(generate_sequences(model, tokenizer, batch_texts, **generation_params))
    data["generated"] = generated
    data.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to pretrained model")
    parser.add_argument("--input_csv", type=str, help="Path to csv with texts")
    parser.add_argument("--input_column", type=str, default="input_text", help="Name of the column with input texts")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to csv to save generated texts, if not "
                                                                     "provided, they will be saved in original csv")
    parser.add_argument("--batch_size", type=int, default=4, help="Size of generation batches")
    parser.add_argument("--max_length", type=int, default=160, help="Parameter for generate() method, maximum length of generated tokens")
    parser.add_argument("--do_sample", type=bool, default=True, help="Parameter for model.generate() method, it enables or disables decoding strategies")
    parser.add_argument("--num_beams", type=int, default=3, help="Parameter for model.generate() method, if set>1, "
                                                                 "it switch model to beam search during text "
                                                                 "generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Parameter for model.generate() method, determines how greedy the generative model is")
    parser.add_argument("--top_k", type=int, default=50, help="Parameter for model.generate() method, the number of "
                                                              "highest probability vocabulary tokens to keep for "
                                                              "top-k-filtering")
    parser.add_argument("--top_p", type=float, default=1.0, help="Parameter for model.generate() method, if set to float < 1, "
                                                                 "only the smallest set of most probable tokens are kept for generation")
    args = parser.parse_args()
    run_test_on_csv(model_path=args.model_path, input_csv=args.input_csv,
                    input_column=args.input_column,
                    output_csv=args.output_csv, batch_size=args.batch_size,
                    max_length=args.max_length, do_sample=args.do_sample,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                    top_k=args.top_k, top_p=args.top_p)