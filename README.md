---

<div align="center">    
 
# Seq2Lightning


</div>
This repository aims to harness the power of Seq2SeqLM transformer models, with a 

Project built to train Seq2SeqLM transformers models with PyTorchLightning.

My main focus here is on T5-based models such as MT5 and FlanT5. These models have shown remarkable multitasking capabilities, a trait that was a defining feature of T5 before it gained mainstream attention in ChatGPT and other large models.

For my small research I've chosen three languages: English, French and Spanish, and two tasks: summarization and translation, but languages can be customized according to someone else's needs.
## Libraries used

- **PyTorchLightning**: powerful framework that simplifies the training and evaluation process of machine learning models
- **Dask**: library that enables parallel and distributed computing for handling larger-than-memory datasets and performing scalable data processing tasks with Pandas-like interface
- **transformers**: amazing library by Hugging Face that provides state-of-the-art pretrained models for various NLP tasks
- **wandb**: machine learning experiment tracking and visualization platform that allows researchers and developers monitor and analyze training runs
## Realized features

- Custom Dataset Building: Gather a comprehensive dataset for translation and summarization tasks, including bilingual summarization. The dataset consists of raw data from the Internet (like WikiMatrix) and datasets from Huggingface library with accordance to chosen languages. Several functions for checking data sample quality were implemented as well.


- Flexible Training Function: Training function has been developed to support various transformers Seq2SeqLM architectures. This function not only facilitates training but also includes logging mechanisms. Additionally, integration with the Weight&Biases dashboard is available as an optional feature, enabling you to conveniently track and manage your experiments.


- Multilanguage Model Cropping: To optimize model size and resource utilization, I implemented the removal of unnecessary languages from the MT5 model, making it more compact and efficient.
## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/kkkravets/Seq2Lightning

# install project   
cd Seq2Lightning
pip install -e .   
 ```   
Now you can run code from console   
 ```bash
# Suppose you want to create new dataset
!python data/loading.py \
                --data_dir /content/data \
                --target_languages fr es \
                --max_rows_per_table=15000 \
                --new_dataset_name=multi_dataset   
```

... or you can explicitly call functions in Python like this:
```python
from src.finetune.train_model import run_training_seq2seq

run_training_seq2seq(
    model_name_or_path=...
)
```
### About Custom dataset creation

*prepare_multilang_dataset* from data module will help you to create you custom dataset, split into train/val/test and save as parquet files.

*preprocess_dataset* function should be called if you want to apply some cleaning before model training.

The example of dataset loading & filtering:
```bash
%cd ./data
!python loading.py --data_dir /content/lp_multi_dataset/raw_data \
                   --target_languages en fr es \
                   --max_rows_per_table=15000 \
                   --new_dataset_name=multi_dataset
                
!python processing.py --dataset_dir="/content/nlp_multi_dataset/raw_data" \
                      --output_dir="/content/nlp_multi_dataset/prepared" \
                      --max_sent_length=160
```

### About Model cropping

Please note that I haven't found a way to automatically load files from Leipzig website. So if you want to reduce MT5 model's vocab, you need:
1. Manually load corpus tar archives from [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download)
2. Specify archives location as well as chosen languages in yaml file, as in the example below. Feel free to just modify the file project_confs.yaml from this repository

```yaml
PROJECT_LANGS:
  - 'en'

LANG_ARCHIVES:
  en: 'leipzig/eng-com_web-public_2018_1M.tar.gz'
```
3. Run *model_utils.py* file from bash or call *cut_t5_based_model* function.

If you are facing an error with sentepiece convertation, try to run the following bash commands and restart the environment:
```bash
pip uninstall protobuf
pip install --no-binary=protobuf protobuf==3.20.3
```
My implementation is heavily based on the David Dale's tutorial: [How to adapt a multilingual T5 model for a single language](https://towardsdatascience.com/how-to-adapt-a-multilingual-t5-model-for-a-single-language-b9f94f3d9c90)
## Plans for the future
- [ ] Write summary or article about my project results
- [ ] Add more data sources and preprocessing function to ensure data quality
- [ ] Add some illustrations of model's inner work to demonstrate how T5 works and what makes it different from other famous models
- [ ] Add instruction tuning feature
- [ ] Introduce more testing functions, allowing users to easily interact with and understand the model's capabilities