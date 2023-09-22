import argparse
import datetime
import json
import os
import re
import shutil
from typing import List, Optional, Union
import urllib
import asyncio
import nltk
import pydantic
import uvicorn
import sys
sys.path.append('../')
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing_extensions import Annotated
from starlette.responses import RedirectResponse
from fastapi.responses import JSONResponse

from chains.local_doc_qa import LocalDocQA, load_vector_store
from configs.model_config import (KB_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN,
                                  VECTOR_SEARCH_SCORE_THRESHOLD, CHUNK_SIZE)

import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import prompt_templete
from utils import torch_gc
import requests
from urllib.parse import urlsplit
from langchain.vectorstores import Milvus
import pymysql

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

from configs.model_config import (KB_ROOT_PATH, EMBEDDING_DEVICE,
                                  EMBEDDING_MODEL, NLTK_DATA_PATH,
                                  VECTOR_SEARCH_TOP_K, LLM_HISTORY_LEN, OPEN_CROSS_DOMAIN,
                                  VECTOR_SEARCH_SCORE_THRESHOLD, CHUNK_SIZE)


def get_folder_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content")


def get_vs_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "vector_store")


def get_file_path(local_doc_id: str, doc_name: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content", doc_name)


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="HTTP status code")
    msg: str = pydantic.Field("success", description="HTTP status message")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


# async def upload_News(
#         knowledge_base_id: str = Body(..., description="知识库名称", example="kb1"),
#         id: str = Body(None, description="资讯id", example=""),  # 资讯id
#         content: str = Body(None, description="资讯正文", example=""),  # 资讯正文
#         title: str = Body(None, description="资讯标题", example=""),  # 资讯标题
#
# ):
#     print('knowledge_base_id: {}'.format(knowledge_base_id))
#     print('id: {}'.format(id))
#     print('title: {}'.format(title))
#     print('content: {}'.format(content))
#     status = "success"
#
#     return BaseResponse(code=200, msg=status)


async def upload_News(
        knowledge_base_id: str = Body(..., description="知识库名称", example="kb1"),
        id: str = Body(None, description="资讯id", example=""),  # 资讯id
        content: str = Body(None, description="资讯正文", example=""),  # 资讯正文
        title: str = Body(None, description="资讯标题", example=""),  # 资讯标题

):
    print('knowledge_base_id: {}'.format(knowledge_base_id))
    print('id: {}'.format(id))
    print('title: {}'.format(title))
    print('content: {}'.format(content))

    def content2txt(title: str, content: str):
        """
        将资讯内容转换为txt 文件，且文件名以title命名
        """
        # 定义文件名，文件名有title的内容组成，但由于文件名需要去除特殊字符，否则编译失败
        # 文件名不能出现这9个字符 / \ : * " < > | ？
        pattern = re.compile('[ /\\:"<>|？?\*]')
        # 基于资讯标题定义file_name
        reg_title = title.replace("&#xa0", "").replace("\n", "").replace("\r", "").strip()
        file_name = str(pattern.sub("-", reg_title)) + ".txt"
        saved_path = get_folder_path(knowledge_base_id)
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        file_path = os.path.join(saved_path, file_name)
        if os.path.exists(file_path) and os.path.getsize(file_path) == len(content):
            file_status = f"文件 {file_name} 已存在。"
            return BaseResponse(code=200, msg=file_status)
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(content)
        except FileNotFoundError or OSError:
            return "资讯入库失败，请检查标题内容是否存在特殊字符后重新尝试！"
        return file_path, file_name

    file_path, file_name = content2txt(title=title, content=content)
    # vs_path 改用miluvs后暂未使用
    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files, _ = local_doc_qa.init_knowledge_vector_store([file_path], vs_path, knowledge_base_id)
    if len(loaded_files) > 0:
        file_status = f"文件 {file_name} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)

app = FastAPI()
app.post("/news/upload_news", response_model=BaseResponse)(upload_News)


def api_start(host, port, my_shared=None, my_local_doc_qa=None):
    global app
    global local_doc_qa
    global shared

    if my_shared:
        shared = my_shared

    if my_local_doc_qa:
        local_doc_qa = my_local_doc_qa


    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # app = FastAPI()
    # app.post("/news/upload_news", response_model=BaseResponse)(upload_News)

    # uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import models.shared as shared
    from models.loader import LoaderCheckPoint
    from models.loader.args import parser
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)

    # 初始化消息
    args = None
    args = parser.parse_args()
    args_dict = vars(args)

    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)

    llm_model_ins = shared.loaderLLM()
    # llm_model_ins.set_history_len(LLM_HISTORY_LEN)

    api_start(args.host, args.port)
    uvicorn.run('test0712_lzc_web_api:app', host=args.host, port=args.port, workers=2)
