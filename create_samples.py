# -*- coding: utf-8 -*-

import xlrd
import os
import random
import numpy as np

from SMOTE import *
#from split import *


from datetime import date, datetime

"""
def read_excel(Paths):
    samples = []
    for p in Paths:
        ExcelPath = p
        ExcelFile = xlrd.open_workbook(r'' + ExcelPath)

        # 获取目标EXCEL文件sheet名
        print(ExcelFile.sheet_names())

        # ------------------------------------

        # 若有多个sheet，则需要指定读取目标sheet例如读取sheet2

        # sheet2_name=ExcelFile.sheet_names()[1]

        # ------------------------------------

        # 获取sheet内容【1.根据sheet索引2.根据sheet名称】

        sheet = ExcelFile.sheet_by_index(0)
        name_index = None
        code_index = None
        year_category=0
        s = os.path.split(ExcelPath)[-1]
        if (s.find('-2') != -1):
            year_category=-2
        elif(s.find('-3')!=-1):
            year_category=-3
        t=sheet.row(0)
        # year_index = None
        for i in range(len(sheet.row(0))):
            if (sheet.row(0)[i].value.find('名称')!=-1):
                name_index = i
            elif (sheet.row(0)[i].value.find('代码')!=-1):
                code_index = i
            # elif ('year' in sheet.row(0)[i].value.encode('utf-8')):
            # year_index = i
        ST=(s.find('normal') == -1)
        for i in range(1, sheet.nrows):
                current_row = sheet.row_values(i)
                current_company = Company()
                current_company.get_name(current_row[name_index])
                current_company.get_year(year_category)
                current_company.get_code(current_row[code_index])
                current_company.get_ST(ST)
                if(current_company.year==-2):
                    current_company.characteristics=sheet.row_values(i)[-7:]
                else:
                    current_company.characteristics=sheet.row_values(i)[-6:]
                samples.append(current_company)
    return samples

"""
def create_samples(paths,smote,smote_index,number_of_not_to_smote):

    features_train=[]
    targets_train=[]
    features_test=[]
    targets_test=[]

    if smote:
        path_to_smote=[]
        path_to_smote.append(paths[smote_index])
        del paths[smote_index]
        sample_pool1 = read_excel(paths)
        sample_pool2=read_excel(path_to_smote)

        samples_train_to_smote = []
        samples_test_smote=[]

        target_flag = False
        for i in range(number_of_not_to_smote):
            r=random.choice(sample_pool2)
            features_test.append(np.array(r.characteristics))
            targets_test.append(int(r.ST))
            samples_test_smote.append(r.characteristics)
            target_flag = r.ST
            sample_pool2.remove(r)
        samples_test_smote=Smote(np.array(samples_test_smote),N=800).over_sampling()
        for i in range(len(samples_test_smote)):
            features_test.append(samples_test_smote[i])
            targets_test.append(int(target_flag))
        samples_test_smote=[]         #清空辅助列表

        for s2 in sample_pool2:
            features_train.append(np.array(s2.characteristics))
            targets_train.append(int(s2.ST))
            target_flag=s2.ST
            samples_train_to_smote.append(s2.characteristics)
        samples_train_to_smote=Smote(np.array(samples_train_to_smote),N=2000).over_sampling()
        for i in range(len(samples_train_to_smote)):
            features_train.append(samples_train_to_smote[i])
            targets_train.append(int(target_flag))

        sub1_sample_pool1, sub2_sample_pool2 =data_split(sample_pool1,0.33,shuffle=True)
        for s1 in sub1_sample_pool1:
            features_test.append(np.array(s1.characteristics))
            targets_test.append(int(s1.ST))
        for s1 in sub2_sample_pool2:
            features_train.append(np.array(s1.characteristics))
            targets_train.append(int(s1.ST))
    else:
        return '相关功能正在开发中'
    return features_train,targets_train,features_test,targets_test
