import os
import argparse
import pandas as pd
import youtokentome as yttm
from sklearn.model_selection import train_test_split

def load_files(path):
    res = ([], [])
    for i, ext in enumerate(['.en', '.ru']):
        with open(path + ext, encoding='utf8') as in_file:
            res[i].extend(in_file.readlines())
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='preparation.py', description='Preprocess the data.')
    parser.add_argument('dataset_path', help='path to the dataset.')
    parser.add_argument('store_path', help='path there to store the results.')
    parser.add_argument('--dataset_size', nargs='?', type=int, help='max number of samples in dataset.')
    parser.add_argument('--vocab_size', nargs='?', type=int, default=2 ** 15,
                    help="the size of the tokenizer's vocabulary.")
    parser.add_argument('--temp_file_path', nargs='?', default='tokenizer_text.temp',
                    help='path there to save the temp file for the tokenizer.')
    
    args = parser.parse_args()
    tokenizer_path = f'{args.dataset_path}_v{args.vocab_size}.tokenizer'

    print('Loading data...')
    data_en, data_ru = load_files(args.dataset_path)
    raw_data = {'English' : [line for line in data_en], 'Russian': [line for line in data_ru]}
    df = pd.DataFrame(raw_data, columns=list(raw_data.keys()))
    # Limit the dataset size
    if args.dataset_size:
        df = df[:args.dataset_size]

    print('Creating train, test, val sets...')
    train, test = train_test_split(df, test_size=0.2)
    test, val = train_test_split(test, test_size=0.5)
    train.to_csv(os.path.join(args.store_path, 'train.csv'), index=False)
    test.to_csv(os.path.join(args.store_path, 'test.csv'), index=False)
    val.to_csv(os.path.join(args.store_path, 'val.csv'), index=False)

    print('Creating tokenizer:')
    with open(args.temp_file_path, 'w', encoding='utf8') as out_file:
        out_file.write('\n'.join(map(str.lower, data_en)))
        out_file.write('\n'.join(map(str.lower, data_ru)))
    # Train tokenizer.
    tokenizer = yttm.BPE.train(data=args.temp_file_path, vocab_size=args.vocab_size, model=tokenizer_path,
                               pad_id=0, unk_id=1, bos_id=2, eos_id=3)
    # Delete temp file.
    os.remove(args.temp_file_path)