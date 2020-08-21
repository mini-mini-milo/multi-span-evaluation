#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:54:26 2020

@author: marie-annexu
"""

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, RobertaForQuestionAnswering, RobertaTokenizer
import torch
import numpy as np
import json

def get_scores(question, answer_text, tokenizer, model):
    encoded_dict = tokenizer(question, answer_text, truncation=True, max_length=512)
    input_ids = encoded_dict['input_ids']
    if 'token_type_ids' in encoded_dict:
        start_scores, end_scores = model(torch.tensor([input_ids]), 
                                         token_type_ids=torch.tensor([encoded_dict['token_type_ids']]))
    else :
        start_scores, end_scores = model(torch.tensor([input_ids]))
    start_scores = start_scores.detach().numpy()[0]
    start_scores[0] = 0
    end_scores = end_scores.detach().numpy()[0]
    end_scores[0] = 0
    return start_scores, end_scores

def add_to_json(keyname, model_results):
    with open('base_models_pred_all.json') as unedited:
        all_models_results = json.load(unedited)
    all_models_results[keyname] = model_results
    with open('base_models_pred_all.json', 'w') as f:
        json.dump(all_models_results, f, indent=4)
        
data = []
with open('multi_dataset.json') as f:
    data = json.load(f)

all_models_results = {}

# models:
# 1. bert-base: twmkn9/bert-base-uncased-squad2
# 2. bert-large: deepset/bert-large-uncased-whole-word-masking-squad2 #works
# 3. roberta-base: deepset/roberta-base-squad2
# 4. roberta-large: ahotrod/roberta_large_squad2
# 5. albert-base: twmkn9/albert-base-v2-squad2 #works
# 6. albert-large: ktrapeznikov/albert-xlarge-v2-squad-v2 #works

model_names = ['twmkn9/bert-base-uncased-squad2', 'deepset/roberta-base-squad2', 'twmkn9/albert-base-v2-squad2']

for model_name in model_names:
    
    print('starting '+model_name)
    
    all_models_results = {}
    model_results = []

    if model_name == 'deepset/roberta-base-squad2':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForQuestionAnswering.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    for triplet in data['multi_dataset']:
        id_true_pred = {}
        id_true_pred['question_id'] = triplet['question_id']
        id_true_pred['true_answers'] = triplet['answers']

        question = triplet['question']
        answer_text = triplet['passage']
        start_scores, end_scores = get_scores(question, answer_text, tokenizer, model)

        id_true_pred['start_scores'] = start_scores.tolist()
        id_true_pred['end_scores'] = end_scores.tolist()
        model_results.append(id_true_pred)
        
    if model_name == 'twmkn9/bert-base-uncased-squad2':
        all_models_results['bert-base'] = model_results
        with open('base_models_pred_all.json', 'w') as f:
            json.dump(all_models_results, f, indent=4)
        print('bert-base done')
    elif model_name == 'deepset/roberta-base-squad2':
        add_to_json('roberta-base', model_results)
        print('roberta-base done')
    elif model_name == 'twmkn9/albert-base-v2-squad2':
        add_to_json('albert-base', model_results)
        print('albert-base done')
