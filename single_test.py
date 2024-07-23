# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Version  : 3.11
# @Author   : Lordera
# @Datetime : 2024/5/22 上午11:46
# @Project  : multi-main
# @File     : single_test.py

import torch
import torch.nn as nn
from model_config import Config
import pickle as pkl
import warnings
import time


# 计算概率化结果
def calculate(out, k):
    out = [p for p in nn.functional.softmax(out, dim=-1)[0].tolist()]
    out_dict = sorted(enumerate(out), key=lambda x: x[1], reverse=True)[:k]
    total = sum([item[1] for item in out_dict])
    predict_class = [(item[0], round(item[1] / total, 4)) for item in out_dict]
    return predict_class


def recognize_label(config, text, vocab, tokenizer, m_dict):
    token = tokenizer(text)
    valid_len = len(token)
    
    if valid_len < config.pad_size:
        token.extend(['<PAD>'] * (config.pad_size - len(token)))
    else:
        token = token[:config.pad_size]
    
    text_input = []
    for word in token:
        text_input.append(vocab.get(word, vocab.get('<UNK>')))
    text_input = [torch.LongTensor(text_input).to(config.device).reshape(1, -1),
                  torch.tensor(valid_len).to(config.device).reshape(1)]
    
    # 第一层场景识别
    text_output = m_dict[10](text_input)
    class_dict_1 = calculate(text_output, k=3)
    predict_layer_1 = class_dict_1
    
    # 第二层意图识别
    predict_layer_2 = []
    for predict_1 in predict_layer_1:
        outputs = m_dict[int(predict_1[0])](text_input)
        class_dict_2 = calculate(outputs, k=3)
        predict_layer_2.extend([(predict_1[0], item[0], round(predict_1[1] * item[1], 4)) for item in class_dict_2])
    predict_layer_2 = sorted(predict_layer_2, key=lambda x: x[2], reverse=True)[:3]
    
    return predict_layer_2


def recognize_item(config, label_result):
    item_result = []
    for item in label_result:
        temp = (config.class_list[int(item[0])],
                config.class_item_list[int(item[0])][int(item[1])], item[2])
        item_result.append(temp)
    return item_result


def test_app(prompt):
    t_config = Config()
    
    text_tokenizer = lambda x: [y for y in x]
    with open('./vocab.pkl', 'rb') as file:
        text_vocab = pkl.load(file)
    with open('./model_dict.pkl', 'rb') as file:
        model_dict = pkl.load(file)
    text_label = recognize_label(t_config, prompt, text_vocab, text_tokenizer, model_dict)
    text_item = recognize_item(t_config, text_label)
    return text_item


def test():
    warnings.filterwarnings('ignore')
    t_config = Config()
    
    text_tokenizer = lambda x: [y for y in x]
    with open('./vocab.pkl', 'rb') as file:
        text_vocab = pkl.load(file)
    with open('./model_dict.pkl', 'rb') as file:
        model_dict = pkl.load(file)
    
    while 1:
        test_text = input('请输入搜索内容<输入0即可退出>: ')
        
        if test_text == '0':
            break
        
        start_time = time.time()
        text_label = recognize_label(t_config, test_text, text_vocab, text_tokenizer, model_dict)
        text_item = recognize_item(t_config, text_label)
        end_time = time.time()
        time_interval = '{:.4f}'.format(end_time - start_time)
        
        print('搜索用时: %s 秒.' % time_interval)
        print(text_label)
        print(text_item)


if __name__ == '__main__':
    test()
