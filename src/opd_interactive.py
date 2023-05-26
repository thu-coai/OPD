# coding=utf-8

"""Inference EVA"""

import os
import random
import numpy as np
import torch
from typing import List
from arguments import get_args
from model import OPDConfig, OPD, OPD_Prompt
from tokenizer import OPDTokenizer

from generation import generate


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


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Placeholder:
    def raise_error(self, *args, **kwargs):
        raise TypeError("You shouldn't access a placeholder's attributes or call it.")

    __getattribute__ = __getitem__ = __call__ = raise_error


def generate_samples(model, tokenizer: OPDTokenizer, args):
    model.eval()

    with torch.no_grad():
        context: List[str] = []
        max_context_length = 30  # TODO: set it in command line arguments
        while True:
            input_text = input("Usr >>> ").strip()
            if not input_text:
                continue
            if input_text == "clear":
                print("Clear Dialog")
                context.clear()
                continue
            elif input_text == "seed":
                seed = int(input("Seed >>> "))
                print("set random seed")
                set_random_seed(seed)
                context.clear()
                continue
            elif input_text.startswith("set len penalty:"):
                args.length_penalty = float(input_text.split(":")[-1].strip())
                print(f"set length penalty to {args.length_penalty}, Clear Dialog")
                set_random_seed(args.seed) # reset rng
                context.clear()
                continue
            elif input_text.startswith("set repetition penalty:"):
                args.repetition_penalty = float(input_text.split(":")[-1].strip())
                print(f"set repetition penalty to {args.repetition_penalty}, Clear Dialog")
                set_random_seed(args.seed) # reset rng  
                context.clear()
                continue
            elif input_text == 'exit':
                exit(0)

            if "[SEP]" in input_text:
                for utt in input_text.split("[SEP]"):
                    context.append(utt)
            else:
                context.append(input_text)
            print('context = ', context)
            # build a dict
            instance = {
                "source": ["".join(t + "\n" for t in context)],
                "target": Placeholder(),  # target is not used.
                "mode": 'lm'
            }
            eos_max = 1
            target_span_len = args.span_length
            eos_num = 0
            # 指定最短生成长度
            min_len = 2 # 确保生成内容不为空
            generated_tokens = []
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
                    generated_tokens.append(it)

            generated_text = ''.join(generated_tokens).strip()
            context.append(generated_text)
            if len(context) > max_context_length:
                n = len(context) - max_context_length
                del context[:n]
            print("Sys >>> {}".format(generated_text))


def initialize():
    # get arguments
    args = get_args()
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def main():
    # Arguments.
    args = initialize()

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # setup tokenizer and model
    tokenizer, model = setup_model(args)

    # setting default batch size to 1
    args.batch_size = 1
    print('Start Inference')
    #generate samples
    generate_samples(model, tokenizer, args)
    

if __name__ == "__main__":
    main()

