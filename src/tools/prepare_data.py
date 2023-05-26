# coding=utf-8

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

from tokenizer import OPDTokenizer
import indexed_dataset

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = OPDTokenizer(self.args.tokenizer_path)

    def convert_to_ids(self, text):
        ids = Encoder.tokenizer.encode(text)
        ids = [j for j in ids if j != Encoder.tokenizer.unk_id]
        return ids
        
    def encode(self, line):
        data = line.strip()
        sents = data.split("\t")

        token_ids = None

        if len(sents) == 1:
            print(sents)
            return None, None, 0
        
        sent_ids = [[i for i in self.tokenizer.encode(sent) if i != Encoder.tokenizer.unk_id] + [self.tokenizer.line_id] for sent in sents]
        
        flag = False
        while len(sent_ids) > 1:
            total_len = sum([len(i) for i in sent_ids])
            if total_len < self.args.max_seq_len: # 成功找到一个
                flag = True
                break
            else:
                sent_ids = sent_ids[1:]
                sents = sents[1:]
        
        if flag is True:
            sep_token = '\n'
            context = ''.join([i + sep_token for i in sents[:-1]])
            response = sents[-1] + sep_token
            return [context], response, len(line)
        else:
            return None, None, 0


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', default="../train_0.txt", type=str, help='Path to input TXT')
    
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="vocab", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--max_seq_len", type=int, default=128, help="max length of a sample")
    group.add_argument("--output_path", default="../data_bin/", type=str)
    group.add_argument('--output_prefix', type=str, help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=32,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args

def main():
    args = get_args()
    startup_start = time.time()
    fin = open(args.input, 'r', encoding='utf-8')
    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)

    print(f"Output prefix: {args.output_prefix}")


    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0


    f = open(os.path.join(args.output_path, "data.txt"), 'w')

    print("Time to startup:", startup_end - startup_start)

    for i, (context, response, bytes_processed) in enumerate(encoded_docs, start=1):
        if context is None:
            continue
        total_bytes_processed += bytes_processed
        
        item = {"source": context, "target": response}
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)
            
    f.close()
    pool.close()

if __name__ == '__main__':
    main()