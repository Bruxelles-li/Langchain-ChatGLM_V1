"""
多个文件上传
"""
import requests
import os
from tqdm import tqdm

url = "http://180.76.177.55:7861/local_doc_qa/upload_files"


file_dir = '/home/python/datacan/pycharm_projects/langchina-ChatGLM/data_preprocess/output_txt'
file_list = os.listdir(file_dir)

payload = {'knowledge_base_id': '审计法规知识'}    # 知识库名称
headers = {
  'Content-Type': 'multipart/form-data'
}
num_of_post = 100   # 单次向接口发送的的文件数
temp = []
for idx, file_name in tqdm(enumerate(file_list)):

    temp.append(('files', (file_name, open(os.path.join(file_dir, file_name), 'rb'), 'text/plain')))
    if (idx + 1) % 100 == 0:
        response = requests.request("POST", url, headers=headers, data=payload, files=temp)
        temp = []
        # print(response.text)

if temp:
    response = requests.request("POST", url, headers=headers, data=payload, files=temp)
    # print(response.text)

