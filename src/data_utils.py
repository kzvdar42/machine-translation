import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import youtokentome as yttm
from functools import partial

class TextDataset(Dataset):

    __output_types = { 'id': yttm.OutputType.ID,
                       'subword':yttm.OutputType.SUBWORD }

    def __init__(self, csv_file, tokenizer, max_len=50, max_len_ratio=1.5):
        self.tokenizer = tokenizer
        df = pd.read_csv(csv_file)
        # Tokenize sentences using tokenizer.
        # TODO: Improve code by rewriting lambdas to smth else.
        tokenize_lambda = lambda x: self.tokenize(x.lower().strip(), 'subword')
        df['eng_enc'] = df.English.apply(tokenize_lambda)
        df['rus_enc'] = df.Russian.apply(tokenize_lambda)
        # Delete sentences that exceed the max length and max length ratio.
        df['en_len'] = df['eng_enc'].str.len()
        df['ru_len'] = df['rus_enc'].str.len()
        df.query(f'ru_len < {max_len} & en_len < {max_len}', inplace=True)
        df.query(f'ru_len < en_len * {max_len_ratio} & ru_len * {max_len_ratio} > en_len', inplace=True)
        # Sort the values for less padding in batching.
        df.sort_values(['ru_len', 'en_len'], ascending=[False, False], inplace=True)
        # TODO: better unpacking
        raw_src, raw_tgt = zip(df[['Russian', 'English']].T.values)
        src, tgt = zip(df[['rus_enc', 'eng_enc']].T.values)
        self.tgt, self.src = tgt[0], src[0]
        self.raw_src, self.raw_tgt = raw_src[0], raw_tgt[0]
        

    def tokenize(self, s, output_type='id'):
        """Tokenize the sentence.
        :param s: the sentence to tokenize
        :param output_type: either 'id' or 'subword' for corresponding output
        :return: tokenized sentence"""
        return self.tokenizer.encode(s, output_type=self.__output_types[output_type],
                                bos=True, eos=True)
    def decode(self, tokens):
        return self.tokenizer.id_to_subword(tokens)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        src = self.src[idx]
        src = [self.tokenizer.subword_to_id(token) for token in src]
        tgt = self.tgt[idx]
        tgt = [self.tokenizer.subword_to_id(token) for token in tgt]
        return src, tgt

def load_datasets(path, tokenizer, ext='.csv'):
    res = []
    for name in  ['train', 'val', 'test']:
        dataset_path = os.path.join(path, name + ext)
        res.append(TextDataset(dataset_path, tokenizer))
    return res


def my_collate(batch, pad_token=0):
    src, tgt = zip(*batch)
    src = [Tensor(s) for s in src]
    tgt = [Tensor(t) for t in tgt]
    # TODO: Generalize padding value
    src = pad_sequence(src, batch_first=True, padding_value=pad_token).long()
    tgt = pad_sequence(tgt, batch_first=True, padding_value=pad_token).long()
    return src.t(), tgt.t()

def make_dataloaders(datasets, batch_size, pad_token, num_workers=0):
    res = []
    for dataset in datasets:
        res.append(DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        collate_fn=partial(my_collate, pad_token=pad_token)))
    return res