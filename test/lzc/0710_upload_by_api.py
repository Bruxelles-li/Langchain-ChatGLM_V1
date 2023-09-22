#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : pro_base_data.py
# @Time    : 2023/5/26 20:40
# @Author  : bruxelles_li
# @Software: PyCharm
import os

import pandas as pd
from tqdm import tqdm
import requests
import json
# 116.63.179.212
url = "http://localhost:7861/news/upload_news"
# http://116.63.179.212:7861/news/upload_news
headers = {
    'Content-Type': 'application/json'
}

import os

# 文件夹路径
folder_path = "/home/zzsn/zhangtao/pycharm_projects/langchain/langchain-ChatGLM/data_preprocess/0809sj-law"

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file_name in tqdm(files):
        file_path = os.path.join(root, file_name)

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # print(content)

        # 提取文件名作为id
        id = os.path.splitext(file_name)[0]
        # print(id)

        # 在这里可以对id和content进行处理，比如保存到字典或写入文件等操作
        payload = json.dumps({
            "knowledge_base_id": "sjdmx_question_database",
            "id": id,
            "title": "",
            "content": content
        })
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            text = response.text
            obj = json.loads(text)
            # print(obj)
            code = obj["code"]
            if code == 200:
                print("success")
            else:
                print("false")
        except:
            continue


#
# df = pd.read_excel('研究中心素材数据/0801待入库数据-5个专题.xlsx', keep_default_na=False).astype(str)
#
# import time
# pre_start_time = time.time()
# for index, row in tqdm(df.iterrows()):
#     # 获取正文、标题、id
#     title = row["title"]
#     content = row["content"]
#     id = row["id"]
#     if len(title) <= 5 or len(content) <= 50 or len(title) >= 50:
#         continue
#     payload = json.dumps({
#         "knowledge_base_id": "yjzx_news_vdb",
#         "id": id,
#         "title": title,
#         "content": content
#     })
#     response = requests.request("POST", url, headers=headers, data=payload)
#     text = response.text
#     obj = json.loads(text)
#     code = obj["code"]
#     if code == 200:
#         print("success")
#     else:
#         print("false")
#     print(f'正在处理第{index+1}篇文章，还剩余{len(df) -index + 1}篇，处理进度为：{round(((index +1)/len(df))*100, 2)}')
# pre_end_time = time.time()
# pre_total_time = round((pre_end_time - pre_start_time), 2)
# print(f'已入库{len(df)}篇，共耗时{pre_total_time}秒')





