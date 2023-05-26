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
        i = json.loads(line)
        task = 0
        
        ids = []
        info = [task]
        control = []

        ids += control
        info.append(len(control))
        if isinstance(i['target'], list): # MLM 多个位置的Mask
            assert len(i['target']) + 1 == len(i['source'])
        elif isinstance(i['target'], str): # LM 结尾的mask
            i['target'] = [i['target']]
        else:
            raise ValueError("invalid i['target'] format")
        i['target'].append('')    

        src_tgt_list = list(zip(i['source'], i['target']))
        for idx, (src, tgt) in enumerate(src_tgt_list):
            src_ids = self.convert_to_ids(src)
            if idx == 0:
                src_ids = [Encoder.tokenizer.bos_id] + src_ids
            
            ids += src_ids
            info.append(len(src_ids))

            tgt_ids = self.convert_to_ids(tgt)
            if idx == len(src_tgt_list) - 1:
                info.extend([len(tgt_ids), 0])
            else:
                info.append(len(tgt_ids))
            
            ids += tgt_ids

        info = info[:1] + np.cumsum(info[1:]).tolist()

        assert len(ids) == info[-1]
        assert len(info) % 2 == 1
        return ids, info, len(line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, help='Path to input TXT')
    
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", type=str)
    group.add_argument('--output_prefix', type=str, help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--uid', type=str, default="00000",
                       help='Number of worker processes to launch')
    group.add_argument('--workers', type=int, default=64,
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

    uid = args.uid
    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 10)

    print(f"Output prefix: {args.output_prefix}")

    context_bin_file = os.path.join(args.output_path, "{}_context_{}.bin".format(args.output_prefix, int(uid)))
    context_idx_file = os.path.join(args.output_path,  "{}_context_{}.idx".format(args.output_prefix, int(uid)))
    target_bin_file = os.path.join(args.output_path,  "{}_target_{}.bin".format(args.output_prefix, int(uid)))
    target_idx_file = os.path.join(args.output_path,  "{}_target_{}.idx".format(args.output_prefix, int(uid)))
    
    builder_context = indexed_dataset.make_builder(context_bin_file, impl=args.dataset_impl, dtype=np.uint16)
    builder_target = indexed_dataset.make_builder(target_bin_file, impl=args.dataset_impl, dtype=np.uint16)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (pair_ids, label_ids, bytes_processed) in enumerate(encoded_docs, start=1):
        if pair_ids is None or label_ids is None:
            continue
        total_bytes_processed += bytes_processed

        # for pids, lids in zip(pair_ids, label_ids):
        builder_context.add_item(torch.IntTensor(pair_ids))
        builder_target.add_item(torch.IntTensor(label_ids))
        
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    builder_context.finalize(context_idx_file)
    builder_target.finalize(target_idx_file)

    pool.close()

if __name__ == '__main__':
    main()