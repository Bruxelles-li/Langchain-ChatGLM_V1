import os
import time

from tqdm import tqdm
from chains.local_doc_qa import LocalDocQA
from configs.model_config import (KB_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN)
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint

# 初始化消息
args = None
args = parser.parse_args()
args_dict = vars(args)
shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
llm_model_ins = shared.loaderLLM()
llm_model_ins.set_history_len(LLM_HISTORY_LEN)

local_doc_qa = LocalDocQA()
local_doc_qa.init_cfg(
    llm_model=llm_model_ins,
    embedding_model=EMBEDDING_MODEL,
    embedding_device=EMBEDDING_DEVICE,
    top_k=VECTOR_SEARCH_TOP_K,
)


def get_folder_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content")


def get_vs_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "vector_store")


def upload_files(file_dir, knowledge_base_id):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    filelist = []
    for file_name in tqdm(os.listdir(file_dir)[:100]):
        file_content = ''
        file_path = os.path.join(saved_path, file_name)
        file_content = open(os.path.join(file_dir, file_name), 'rb').read()
        if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
            continue

        with open(file_path, "ab+") as f:
            f.write(file_content)

        filelist.append(file_path)
    if filelist:
        print('开始入向量库 ...')
        time_start = time.time()
        vs_path, loaded_files, _ = local_doc_qa.init_knowledge_vector_store(filepath=filelist,
                                                                            vs_path=get_vs_path(knowledge_base_id),
                                                                            knowledge_base_id=knowledge_base_id)
        # if len(loaded_files):
        #     file_status = f"已上传 {'、'.join([os.path.split(i)[-1] for i in loaded_files])} 至知识库，并已加载知识库，请开始提问"
        #     print('{}个文件已入库！'.format(len(filelist)))
        #     return '{}个文件已入库！'.format(len(filelist))

        time_cost = time.time() - time_start
        print('{}个文件已入库！用时：{} s'.format(len(filelist), round(time_cost, 2)))
    else:
        print("文件未成功加载，请重新上传文件")
        # file_status = "文件未成功加载，请重新上传文件"
        # return file_status


if __name__ == "__main__":

    upload_files(file_dir='/home/zzsn/zhangtao/pycharm_projects/langchain/langchain-ChatGLM/data_preprocess/0815-yjzx-test',
                 knowledge_base_id='0815_test_yjzxzt_database')
    # upload_files(file_dir='/home/python/datacan/pycharm_projects/langchina-ChatGLM/data_preprocess/0602-研究中心待入库数据/yjzxsc_2',
    #              knowledge_base_id='国资大数据')

    #


