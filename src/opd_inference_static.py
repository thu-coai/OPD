3#coding:utf-8

import time
from tqdm import tqdm
import random
import torch
import numpy as np
import os
import json
from model import OPDConfig, OPD, OPD_Prompt
from tokenizer import OPDTokenizer

from arguments import get_args
from generation import generate, calculate_ppl

def get_tokenizer(args):
    if args.use_line_token_as_eos:
        tokenizer = OPDTokenizer(args.vocab_file, space_token = '</_>', line_token = '</n>', eos_token='</n>')
    else:
        tokenizer = OPDTokenizer(args.vocab_file, space_token = '</_>', line_token = '</n>')
    return tokenizer

def get_model(args, vocab_size):
    config = OPDConfig.from_json_file(args.model_config)
    # config.vocab_size = vocab_size
    print ("vocab size:%d"%(vocab_size))

    if args.prompt_infer:
        model = OPD_Prompt(config)
    else:
        model = OPD(config)
    model.cuda()
    model.load_state_dict(
        torch.load(args.load),
        strict = True
    )
    return model


def setup_model(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer.vocab_size)
    print("Model mem\n", torch.cuda.memory_summary())
    return tokenizer, model

def initialize():
    args = get_args()
    return args

def main():
    args = initialize()
    
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer, model = setup_model(args)
    
    fout = open("{}".format(args.output_file), "w", encoding="utf-8")
    fin = open('{}'.format(args.input_file), 'r', encoding='utf-8')
    
    model.eval()

    with torch.no_grad():
        for line in tqdm(fin):
            utts = line.strip().split('\t')
            instance = {}
            instance['source'] = ['\n'.join(utts[:-1]) + '\n']
            instance['target'] = utts[-1] + '\n'
            instance['mode'] = 'lm'
            
            if instance['mode'] == 'lm':
                eos_max = 1
            else:
                eos_max = 2
            
            target_span_len = args.span_length
            # 每个instance指定不同的target span长度
            # target_span_len = int(len(tokenizer.encode(instance['source'][0]))*0.45)

            eos_num = 0
    
            min_len = 2
            if os.environ.get("DEBUG"):
                fout.write("target: " + instance['target'].strip() + "\n")
                fout.write("pred: ")
            for it in generate(model, tokenizer, instance, target_span_len, beam=args.beam_size,
                                temperature = args.temperature, top_k = args.top_k, top_p = args.top_p,
                                no_repeat_ngram_size = args.no_repeat_ngram_size, repetition_penalty = args.repetition_penalty, 
                                random_sample=args.random_sample, min_len=min_len, contrastive_search=args.use_contrastive_search,
                                length_penalty=args.length_penalty, prompt_length=model.prompt_length):
                
                if eos_num == eos_max:
                    break

                if it == '</s>' or it == '\n':
                    eos_num += 1
                
                if it != '\n':
                    fout.write(it)
                fout.flush()
            fout.write('\n')

        fin.close()
        fout.close()

if __name__ == "__main__":
    main()
