"""
单个文件上传
"""

import requests
import os
from tqdm import tqdm

url = "http://127.0.0.1:7861/local_doc_qa/upload_file"


file_dir = '/home/python/datacan/pycharm_projects/langchina-ChatGLM/data_preprocess/output_txt'
file_list = os.listdir(file_dir)

payload = {'knowledge_base_id': '审计法规知识'}  # 知识库名称
headers = {}
for file_name in tqdm(file_list):
    files = [
        ('file', (file_name, open(os.path.join(file_dir, file_name), 'rb'), 'text/plain'))
    ]
    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
