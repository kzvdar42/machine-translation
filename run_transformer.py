from nltk.translate.bleu_score import corpus_bleu
import os
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import youtokentome as yttm

from src.data_utils import load_datasets, make_dataloaders
from src.Transformer import TransformerModel

def run_model(model, criterion, optimizer, data_iterator, is_train_phase, n_words=1, desc=''):
    """Run one epoch of a model with given data.
    
    :param model: model to run on
    :param criterion: critetion to use
    :param optimizer: optimizer to use
    :param data_iterator: iterator of (x, y) data tuples
    :param is_train_phase: True if you want to train
    :param n_words: number of words to predict, the bigger the longer it takes to run
    :param desc: description for tqdm bar
    :return: epoch loss
    """
    if is_train_phase:
        model.train() # Turn on the train mode
    else:
        model.eval()
    total_loss = 0.0
    pbar = tqdm(total=len(data_iterator), desc=desc)
    for i, (src, tgt) in enumerate(data_iterator):
        src, tgt = src.to(device), tgt.to(device)
        
        tgt_losses = 0.0
        # Predict `n_words` last words.
        for j in range(max(1, len(tgt) - n_words), len(tgt)):
            optimizer.zero_grad()
            tgt_in = tgt[:j, :]
            tgt_out = tgt[1:j+1, :]
            
            with torch.set_grad_enabled(is_train_phase):
                output = model(src, tgt_in).transpose(1, 2)
                loss = criterion(output, tgt_out)

                if is_train_phase:
                    loss.backward()
                    # Clip gradient to deal with gradient explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
            tgt_losses += loss.item()
        total_loss += tgt_losses / j
        pbar.update(1)
        pbar.set_description(desc + f'- loss: {total_loss / (i+1):7.4}')
    return total_loss / (i+1)

def train_model(model, n_epochs, data_iterators,
                criterion, optimizer, scheduler=None, n_words=1, model_save_path=None):
    stats = {'train':{'loss':[]},
             'val':{'loss':[]}}
    best_loss = None
    
    for epoch in range(n_epochs):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        tqdm.write(f'------------ Epoch {epoch}; lr: {lr:.5f} ------------')
        for phase in ['train', 'val']:
            desc = f'{phase.title()} Epoch #{epoch} '
            epoch_loss = run_model(model, criterion, optimizer,
                                   data_iterators[phase], phase == 'train',
                                   n_words, desc)
            stats[phase]['loss'].append(epoch_loss)
            print_hist = lambda l: ' -> '.join(map(lambda x:f"{x:.4}", l[-2:]))
            tqdm.write(f'{phase.title()} Loss: ' + print_hist(stats[phase]['loss']))
        if best_loss == None or stats['val']['loss'][-1] < best_loss:
            best_loss = stats['val']['loss'][-1]
            tqdm.write('Smallest val loss')
            tqdm.write('Saving model...')
            if model_save_path:
                try:
                    torch.save(model, model_save_path)
                    tqdm.write('Saved successfully')
                except FileNotFoundError:
                    tqdm.write('Error during saving!')
        try:
            translate(model, tokenizer, 'Машинное обучение это здорово!', verbose=True)
            rand_ind = np.random.randint(0, len(data_iterators['test']))
            translate(model, tokenizer, data_iterators['test'].raw_src[rand_ind], verbose=True)
        except:
            tqdm.write('Error while translation.')
        if scheduler:
            scheduler.step()
    return stats


def subword_to_str(tokens):
    return ''.join(tokens).replace('▁', ' ')

def tokens_to_str(tokenizer, tokens):
    return subword_to_str([tokenizer.id_to_subword(ix) for ix in tokens])

def translate(model, tokenizer, text, max_len=80, verbose=False):
    model.eval()
    # Get the device the model is stored on.
    device = next(model.parameters()).device
    
    if verbose:
        print('------------ Translation ------------')
        print('Input:', text)
    # Prepare text
    src = tokenizer.encode(text, output_type=yttm.OutputType.ID,
                           bos=True, eos=True)
    src = Tensor(src).long().to(device)
    # Run encoder
    src_enc, src_mask, _ = model.preprocess(src, 'src')
    e_outputs = model.transformer.encoder(src_enc, 
                                          src_mask,
                                          )
    
    # Prepare tensor for answers
    outputs = torch.zeros(max_len).type_as(src.data)
    # Set the first token as '<sos>'
    outputs[0] = torch.LongTensor([BOS_TOKEN])
    vals = []
    for i in range(1, max_len):
        outputs_enc, tgt_mask, _ = model.preprocess(outputs[:i].unsqueeze(1), 'tgt')
#         memory_mask = model._generate_square_subsequent_mask(len(src), i+1).to(src.device)
        d_out = model.transformer.decoder(outputs_enc, e_outputs,
                                          tgt_mask=tgt_mask,
#                                           memory_mask=memory_mask,
                                          )
        out = model.out(d_out)
        out = F.softmax(out, dim=-1)
        val, ix = out.data.topk(3, dim=-1)
        outputs[i] = ix[-1][0][0]
        if outputs[i] == EOS_TOKEN:
            break
    result = tokens_to_str(tokenizer, outputs[:i+1])
    if verbose:
        print('Output weights:')
        for j in range(min(3, i)):
            print(f'  {j}', {tokenizer.id_to_subword(k):v.item()
                             for k, v in zip(ix[j][0], val[j][0])})
        print('translation:', result)
    return result


def translate_beam(model, tokenizer, text, max_len=10, beam_capacity=3, verbose=False):
    """
    Algorithm: https://www.youtube.com/watch?v=RLWuzLLSIgw
    """
    model.eval()
    # Get the device the model is stored on.
    device = next(model.parameters()).device
    if verbose:
        print('------------ Translation ------------')
        print('Input:', text)
    # Prepare text
    src = tokenizer.encode(text, output_type=yttm.OutputType.ID,
                           bos=True, eos=True)
    src = Tensor(src).long().to(device)
    # Run encoder
    src_enc, src_mask, _ = model.preprocess(src, 'src')
    e_outputs = model.transformer.encoder(src_enc, 
#                                           src_mask,
                                          )

    # Prepare tensor for answers
    basic_vec = torch.zeros(max_len).type_as(src.data)
    basic_vec[0] = torch.LongTensor([BOS_TOKEN])

    beam_pool = [(basic_vec, 1.0)]

    def beam_filter(pool, top_k=beam_capacity):
        return sorted(pool, key=lambda x: x[1], reverse=True)[:top_k]

    for i in range(1, max_len):
        if verbose:
            print("Beam epoch: ", i)
        new_pool = []
        # For each candidate path:
        for beam, old_prob in beam_pool:
            outputs_enc, tgt_mask, _ = model.preprocess(beam[:i].unsqueeze(1), 'tgt')
            d_out = model.transformer.decoder(outputs_enc, e_outputs,
                                              tgt_mask=tgt_mask,
                                              )
            out = model.out(d_out).cpu().detach()
            out = F.softmax(out, dim=-1)
            probs, ixs = out[-1, :].topk(beam_capacity)
            for prob, token_id in zip(probs.squeeze(), ixs.squeeze()):
                tmp_beam = beam.clone()
                tmp_beam[i] = token_id.item()
                new_pool.append((tmp_beam, prob * old_prob))
        beam_pool = beam_filter(new_pool)
        if verbose:
            for beam, old_prob in beam_pool:
                print("Candidate '{}' with prob: {:.7f}".format(
                    tokens_to_str(tokenizer, beam[1:i + 1]), prob * old_prob
                ))
        # Stop if EOS_TOKEN
        if beam_pool[0][0][i] == EOS_TOKEN:
            break
    the_best = beam_filter(beam_pool, 1)[0][0]
    # Cut by EOS_TOKEN
#     if EOS_TOKEN in the_best:
#         i = (the_best == EOS_TOKEN).nonzero()[0]
    result = tokens_to_str(tokenizer, the_best[:i+1])
    return result


def calc_BLEU(model, data):
    references, candidates = [], []
    pbar = tqdm(total=len(data), desc='Test BLEU score')
    # TODO: batch translation.
    for raw_src, raw_tgt in zip(data.raw_src, data.raw_tgt):
        references.append([raw_tgt])
        candidate = translate(model, tokenizer, raw_src)
        candidate = candidate.replace('<BOS>', '').replace('<EOS>', '')
        candidates.append(candidate)
        pbar.update(1)
    score = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    print(f'Test BLEU score - {score:.4f}')


PADDING_TOKEN = 0
UNK_TOKEN = 1
BOS_TOKEN = 2
EOS_TOKEN = 3

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='run_transformer.py', description='Run the transformer.')
    parser.add_argument('data_path', help='path to the train/test/val sets.')
    parser.add_argument('n_epochs', type=int, help='number of training epochs.')
    parser.add_argument('tokenizer_path', help='path to the tokenizer.')
    parser.add_argument('--model_save_path', nargs='?', help="there to load/save the model.")
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help="batch size for training/validation.")
    parser.add_argument('--learning_rate', nargs='?', type=float, default=0.1,
                        help="learning rate for training.")
    parser.add_argument('--n_words', nargs='?', type=int, default=1,
                        help="number of words to train on.")
    parser.add_argument('--emb_size', nargs='?', type=int, default=512,
                        help='embedding dimension.')
    parser.add_argument('--n_hid', nargs='?', type=int, default=2048,
                        help='the dimension of the feedforward network model in nn.TransformerEncoder.')
    parser.add_argument('--n_layers', nargs='?', type=int, default=6,
                        help='the number of encoder/decoder layers in transformer.')
    parser.add_argument('--n_head', nargs='?', type=int, default=8,
                        help='the number of heads in the multiheadattention layers.')
    parser.add_argument('--dropout', nargs='?', type=float, default=0.1,
                        help='dropout rate during the training.')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer.
    tokenizer = yttm.BPE(model=args.tokenizer_path)

    ntokens_src = tokenizer.vocab_size()
    ntokens_tgt = tokenizer.vocab_size()

    # Load data.
    print('Loading data...')
    train_data, val_data, test_data = load_datasets(args.data_path, tokenizer)
    print('Train set len:', len(train_data),
        '\nVal set len:', len(val_data),
        '\nTest set len:', len(test_data))

    # Make dataloaders.
    print('Making dataloaders...')
    (train_iterator,
    val_iterator) = make_dataloaders([train_data, val_data],
                                    batch_size=args.batch_size,
                                    pad_token=PADDING_TOKEN,
                                    num_workers=0)

    data_iterators = {
        'train': train_iterator,
        'val': val_iterator,
        'test': test_data,
    }

    if args.model_save_path and os.path.exists(args.model_save_path):
        print('Loading saved model...')
        model = torch.load(args.model_save_path)
    else:
        print('Creating model...')
        model = TransformerModel(ntokens_src, ntokens_tgt,
                                 args.emb_size, args.n_head, args.n_hid,
                                 args.n_layers, PADDING_TOKEN, args.dropout).to(device)

    # Ignore padding index during the loss computation.
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_TOKEN, reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3.0, gamma=0.95)
    stats = train_model(model, args.n_epochs, data_iterators,
                        criterion, optimizer, scheduler, args.n_words, args.model_save_path)
    
    # Calculate result score.
    calc_BLEU(model, test_data)