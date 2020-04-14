# Machine Translation
Russian to English machine translation using Transformer model.


## Purpose
This project was done during the *Practical Machine Learning and Deep Learning* course at Spring 2020 semester at *Innopolis University* (Innopolis, Russia).


## How to run
Firstly install all requirements from `requirements.txt`.

#### Data preparation
Then you need to prepare your dataset for the model, running the `preparation.py`:
```
usage: preparation.py [-h] [--dataset_size [DATASET_SIZE]]
                      [--vocab_size [VOCAB_SIZE]]
                      [--temp_file_path [TEMP_FILE_PATH]]
                      dataset_path store_path

positional arguments:
  dataset_path          path to the dataset.
  store_path            path there to store the results.

optional arguments:
  -h, --help            show this help message and exit
  --dataset_size [DATASET_SIZE]
                        max number of samples in dataset.
  --vocab_size [VOCAB_SIZE]
                        the size of the tokenizer's vocabulary.
  --temp_file_path [TEMP_FILE_PATH]
                        path there to save the temp file for the tokenizer.
```
It will create the train/test/val datasets and a BPE tokenizer for the model.

#### Training
After that you can train the model using the `run_transformer.py`:
```
usage: run_transformer.py [-h] [--model_save_path [MODEL_SAVE_PATH]]
                          [--batch_size [BATCH_SIZE]]
                          [--learning_rate [LEARNING_RATE]]
                          [--n_words [N_WORDS]] [--emb_size [EMB_SIZE]]
                          [--n_hid [N_HID]] [--n_layers [N_LAYERS]]
                          [--n_head [N_HEAD]] [--dropout [DROPOUT]]
                          data_path n_epochs tokenizer_path

positional arguments:
  data_path             path to the train/test/val sets.
  n_epochs              number of training epochs.
  tokenizer_path        path to the tokenizer.

optional arguments:
  -h, --help            show this help message and exit
  --model_save_path [MODEL_SAVE_PATH]
                        there to load/save the model.
  --batch_size [BATCH_SIZE]
                        batch size for training/validation.
  --learning_rate [LEARNING_RATE]
                        learning rate for training.
  --n_words [N_WORDS]   number of words to train on.
  --emb_size [EMB_SIZE]
                        embedding dimension.
  --n_hid [N_HID]       the dimension of the feedforward network model in
                        nn.TransformerEncoder.
  --n_layers [N_LAYERS]
                        the number of encoder/decoder layers in transformer.
  --n_head [N_HEAD]     the number of heads in the multiheadattention layers.
  --dropout [DROPOUT]   dropout rate during the training.
```
After the training it will compute the BLEU score on test dataset.

#### Translation
To translate the text use the `translate.py`:
```
usage: translate.py [-h] [--encoding [ENCODING]] [--max_len [MAX_LEN]]
                    model_path tokenizer_path in_data_path out_data_path

positional arguments:
  model_path            path to the trained model.
  tokenizer_path        path to the tokenizer.
  in_data_path          path to the input data.
  out_data_path         path where to save the results.

optional arguments:
  -h, --help            show this help message and exit
  --encoding [ENCODING]
                        encoding for files.
  --max_len [MAX_LEN]   maximum translation length.
```
