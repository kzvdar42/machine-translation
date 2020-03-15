import os
from tqdm import tqdm
import torch
import argparse
import youtokentome as yttm

from src.Transformer import TransformerModel, PositionalEncoding
from run_transformer import translate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='translate.py', description='Translate given text.')
    parser.add_argument('model_path', help='path to the trained model.')
    parser.add_argument('tokenizer_path', help='path to the tokenizer.')
    parser.add_argument('in_data_path', help='path to the input data.')
    parser.add_argument('out_data_path', help='path where to save the results.')
    parser.add_argument('--encoding', nargs='?', default='utf8', help='encoding for files.')
    parser.add_argument('--max_len', nargs='?', type=int, default=50, help='maximum translation length.')
    args = parser.parse_args()

    # Create out directory if it doesn't exist.
    os.makedirs(os.path.dirname(args.out_data_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer.
    tokenizer = yttm.BPE(model=args.tokenizer_path)
    # Load model.
    model = torch.load(args.model_path).to(device)
    
    # Load input data.
    with open(args.in_data_path, encoding=args.encoding) as in_file:
        input_sentences = in_file.readlines()
    
    # Translation.
    pbar = tqdm(desc='Translation', total=len(input_sentences))
    with open(args.out_data_path, 'w', encoding=args.encoding) as out_file:
        for sentence in input_sentences:
            translation = translate(model, tokenizer, sentence, max_len=args.max_len)
            translation = translation.replace('<BOS>', '').replace('<EOS>', '').strip()
            out_file.write(translation + '\n')
            pbar.update(1)

