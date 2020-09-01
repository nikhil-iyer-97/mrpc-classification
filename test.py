from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)

from pytorch_transformers import AdamW

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)

ALL_MODELS = tuple(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

def evaluate(args, model, tokenizer, input1, input2, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args["task_name"]
    input_ids, attention_mask, token_type_ids = load_and_cache_examples(args, eval_task, tokenizer, input1, input2, evaluate=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    model.eval()

    with torch.no_grad():
        inputs = {'input_ids':      input_ids.to(args["device"]),
                    'attention_mask': attention_mask.to(args["device"]),
                    'token_type_ids': token_type_ids.to(args["device"])}
        
        outputs = model(**inputs)[0].argmax().item()
        return "sentence_1 is a paraphrase of sentence_2\n" if outputs else "sentence 1 is not a paraphrase of sentence_2\n"
        
def load_and_cache_examples(args, task, tokenizer, input1, input2, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    label_list = processor.get_labels()
    
    examples = processor._create_example_for_test(input1, input2) 
    
    features = convert_examples_to_features(examples, label_list, args["max_seq_length"], tokenizer, output_mode,
        cls_token_at_end=False,            
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=False,           
        pad_on_left=False,               
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    return all_input_ids, all_input_mask, all_segment_ids

def test_example(line1, line2):
    # Setup CUDA, GPU & distributed training
    
    args = json.load(open("test_args.json", "r"))

    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")    
    args["device"] = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args["local_rank"] in [-1, 0] else logging.WARN)

    # Prepare GLUE task
    if args["task_name"] not in processors:
        raise ValueError("Task not found: %s" % (args["task_name"]))
    processor = processors[args["task_name"]]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    
    _, model_class, tokenizer_class = MODEL_CLASSES[args["model_type"]]
    
    model = model_class.from_pretrained(args["ckpt_dir"])
    tokenizer = tokenizer_class.from_pretrained(args["ckpt_dir"], do_lower_case=True)
    model.to(args["device"])

    # Evaluation
    answer = evaluate(args, model, tokenizer, line1, line2)
    return answer

