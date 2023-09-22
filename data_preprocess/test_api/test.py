import requests

url = "http://180.76.177.55:7861/local_doc_qa/upload_files"

payload = {'knowledge_base_id': '审计法规知识'}
files = [
    ('files', ('“211工程”专项资金管理暂行办法.txt', open(r'../output_txt/“211工程”专项资金管理暂行办法.txt', 'rb'), 'text/plain')),
    ('files', ('“科技兴农计划”资金管理办法（试行）.txt', open(r'../output_txt/“科技兴农计划”资金管理办法（试行）.txt', 'rb'), 'text/plain'))
]
headers = {
    'Content-type': 'multipart/form-data'
}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)
