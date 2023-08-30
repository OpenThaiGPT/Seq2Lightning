import csv
import gzip
import langid
from typing import List, Optional


def convert_tsv_to_csv(input_file: str, output_file: str,
                       header_row: Optional[List[str]] = None,
                       max_rows: int = 500000,
                       threshold: float = 1.04) -> None:
    """Read the gzipped TSV file and save it as CSV"""
    with gzip.open(input_file, 'rt', encoding='utf-8') as tsv_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            if header_row is not None:
                csv_writer.writerow(header_row)
            for i, line in enumerate(tsv_file):
                fields = line.strip().split('\t')
                if len(fields) != 3:
                    continue
                if threshold is not None and float(fields[0]) < threshold:
                    continue
                csv_writer.writerow(fields)
                if i == max_rows:
                    break


def check_language(input_string: str, assumed_lang: str) -> bool:
    """Check if stated language matches predicted"""
    real_lang = langid.classify(input_string[:150])[0]
    return real_lang == assumed_lang