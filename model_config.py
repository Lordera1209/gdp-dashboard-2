#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Version  : 3.11
# @Author   : Lordera
# @Datetime : 2024/7/4 下午3:00
# @Project  : classify_box_1
# @File     : model_config.py

import torch


class Config(object):
    def __init__(self):
        # <场景意图分类>
        self.class_list = ['开学', '新授课', '单元复习', '月考', '期中', '期末', '小升初', '中考', '高考', '合格考']
        self.class_list_0 = ['开班会', '摸底考', '学期教学计划']
        self.class_list_1 = ['写教案', '准备课件', '单元解读', '准备学案', '作业']
        self.class_list_2 = ['知识清单', '单元练习']
        self.class_list_3 = ['知识点、章节复习', '模拟卷', '正式考']
        self.class_list_4 = ['知识点、章节复习', '真题汇编', '模拟卷', '正式考']
        self.class_list_5 = ['知识点、章节复习', '真题汇编', '模拟卷', '正式考']
        self.class_list_6 = ['知识点复习', '模拟卷', '真题汇编', '正式考']
        self.class_list_7 = ['知识点复习', '真题汇编', '一模模拟', '二模模拟', '三模模拟', '押题', '正式考']
        self.class_list_8 = ['知识点复习', '真题汇编', '一模模拟', '二模模拟', '三模模拟', '押题', '正式考']
        self.class_list_9 = ['知识点复习', '真题汇编', '模拟卷', '正式卷']

        self.class_item_list = [self.class_list_0, self.class_list_1, self.class_list_2, self.class_list_3,
                                self.class_list_4, self.class_list_5, self.class_list_6, self.class_list_7,
                                self.class_list_8, self.class_list_9]

        self.count_dict = {0: 0, 1: 3, 2: 8, 3: 10, 4: 13, 5: 17, 6: 21, 7: 25, 8: 32, 9: 39}

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.pad_size = 32
