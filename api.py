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
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
import prompt_templete
from utils import torch_gc
import requests
from urllib.parse import urlsplit
from langchain.vectorstores import Milvus
import pymysql

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

mysql_conn = pymysql.connect(host='114.115.185.13',
                             port=3305,
                             user='root',
                             password='sc24&bgqsc',
                             database='clb_project')


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


class ListDocsResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of document names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class SummaryResponse(BaseResponse):
    summary: str = pydantic.Field(..., description="返回的摘要")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "summary": "这是文本摘要",
            }
        }


class ChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")
    response: str = pydantic.Field(..., description="Response text")
    history: List[List[str]] = pydantic.Field(..., description="History text")
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }


class LLM_Params(BaseModel):
    max_length: Optional[int] = 2048
    top_p: Optional[float] = 0.70
    temperature: Optional[float] = 0.95


def get_folder_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content")


def get_vs_path(local_doc_id: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "vector_store")


def get_file_path(local_doc_id: str, doc_name: str):
    return os.path.join(KB_ROOT_PATH, local_doc_id, "content", doc_name)


async def extract_contract_info(
        file: UploadFile = File(description="A single binary file"),
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file.filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file.filename} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)

    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
    if len(loaded_files) > 0:
        file_status = f"文件 {file.filename} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)

    if file_path.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(file_path, mode="elements")
        docs = loader.load()
        res = ''
        for d in docs:
            res += ' '
            res += d.page_content
        res = res.strip()
        print(res)

    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)


async def xiansuo_upload_file(
        file: UploadFile = File(description="A single binary file"),
        database_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    knowledge_base_id = 'xian_suo_finder_vdb{}'.format(database_id)
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file.filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file.filename} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)

    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files, _ = local_doc_qa.init_knowledge_vector_store([file_path],
                                                                        vs_path,
                                                                        knowledge_base_id)
    if len(loaded_files) > 0:
        file_status = f"文件 {file.filename} 已上传至新的知识库，并已加载知识库，请开始提问。"
        # return BaseResponse(code=200, msg=file_status)
        # loader = UnstructuredPaddlePDFLoader(file_path)
        # docs = loader.load()
        # doc_list = []
        # for doc in docs:
        #     doc_list.append(doc.page_content)
        ocr_file_path = os.path.join(saved_path, 'tmp_files', '{}.txt'.format(file.filename))
        print('ocr_file_path: {}'.format(ocr_file_path))
        doc = open(ocr_file_path, "r", encoding='utf-8').read()

        return JSONResponse({'code': 200, 'msg': file_status, 'doc': doc})
    else:
        file_status = "文件上传失败，请重新上传"
        # return BaseResponse(code=500, msg=file_status)
        return JSONResponse({'code': 500, 'msg': file_status})


async def xiansuo_stream_chat(websocket: WebSocket):
    """
    合同知识库
    """
    await websocket.accept()
    turn = 1
    while True:
        input_json = await websocket.receive_json()
        database_id = input_json.get("database_id", '')
        question = input_json.get("question", "")
        history = input_json.get("history", [])
        max_length = input_json.get("max_length", 2048)
        top_p = input_json.get("top_p", 0.70)
        temperature = input_json.get("temperature", 0.10)
        vector_search_top_k = input_json.get("vector_search_top_k", 15)

        # await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        try:
            vector_store = Milvus(
                embedding_function=local_doc_qa.embeddings,
                collection_name='xian_suo_finder_vdb{}'.format(database_id),
                connection_args={"host": "127.0.0.1", "port": "19530"},
                search_params={"HNSW": {"metric_type": "L2", "params": {"ef": vector_search_top_k}}},
            )
        except Exception as e:
            await websocket.send_text('Error: {}'.format(e))

        query = question
        streaming = True
        # history = []  # 单轮问答，这样可以避免字符超限制，目前支持最大2048tokens
        last_print_len = 0
        local_doc_qa.top_k = vector_search_top_k
        for resp, history in local_doc_qa.get_knowledge_based_answer0525(
                vector_store=vector_store,
                my_llm=shared.loaderCheckPoint,
                query=query,
                chat_history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                streaming=streaming):
            # print('resp: {}'.format(resp))
            answer = resp['response'][last_print_len:]
            # encoded_answer = answer.encode('utf-8')
            await websocket.send_text(answer)
            last_print_len = len(resp['response'])

        source = "<br> <br>"
        # source += "".join(
        #     [
        #         f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
        #         f"""{doc.page_content}\n"""
        #         f"""</details>"""
        #         for i, doc in
        #         enumerate(resp["source_documents"])])
        source += "".join(
            [
                f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc[0].metadata["source"])[-1]}</summary>\n"""
                f"""{doc[0].page_content}\n"""
                f"""</details>"""
                for i, doc in
                enumerate(resp["source_documents"])])

        await websocket.send_text(source)
        # history[-1][-1] += source
        # result = history[-1][-1]
        # yield history, ""

        torch_gc()

        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                },
                ensure_ascii=False,
            )
        )
        turn += 1


async def upload_book(
        knowledge_base_id: str = Body(..., description="知识库名称", example="kb1"),
        id: str = Body(None, description="书籍名称", example=""),  # 书籍id
        name: str = Body(None, description="书籍名称", example=""),  # 书籍名称
        no_isbn: str = Body(None, description="ISBN号", example=""),  # ISBN号
        site_isbn: str = Body(None, description="网站ISBN号", example=""),  # 网站ISBN号
        author: str = Body(None, description="作者", example=""),  # 作者
        publishing_house: str = Body(None, description="出版社", example=""),  # 出版社
        publish_date: str = Body(None, description="发布日期(年度)", example=""),  # 发布日期(年度)
        page_size: str = Body(None, description="页数", example=""),  # 页数
        clc_type_id: str = Body(None, description="中图法分类号", example=""),  # 中图法分类号
        clc_type_name: str = Body(None, description="中图分类号对应的类别名称", example=""),  # 中图分类号对应的类别名称
        create_by: str = Body(None, description="创建人", example=""),  # 创建人
        create_time: str = Body(None, description="创建时间", example=""),  # 创建时间
        filePath: str = Body(None, description="图书文件地址", example="")  # 图书文件地址

):
    print('knowledge_base_id: {}'.format(knowledge_base_id))
    print('id: {}'.format(id))
    print('name: {}'.format(name))
    print('no_isbn: {}'.format(no_isbn))
    print('site_isbn: {}'.format(site_isbn))
    print('author: {}'.format(author))
    print('publishing_house: {}'.format(publishing_house))
    print('publish_date: {}'.format(publish_date))
    print('page_size: {}'.format(page_size))
    print('clc_type_id: {}'.format(clc_type_id))
    print('clc_type_name: {}'.format(clc_type_name))
    print('create_by: {}'.format(create_by))
    print('create_time: {}'.format(create_time))
    print('filePath: {}'.format(filePath))

    def file_download(url):
        """
        下载文件
        """
        # url = 'https://www.guizhou.gov.cn/zwgk/rsxx/sydwgkzp/202306/P020230627609639286577.docx'

        split_result = urlsplit(url)
        path = split_result.path
        filename = path.split("/")[-1]

        response = requests.get(url)

        if response.status_code == 200:
            # with open('file.pdf', 'wb') as f:
            #     f.write(response.content)
            print('文件下载成功')
            filecontent = response.content
            return filecontent, filename
        else:
            print('下载失败，状态码为', response.status_code)

            return None

    file_content, file_name = file_download(url=filePath)

    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    # file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file_name)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file_name} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)

    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files, _ = local_doc_qa.init_knowledge_vector_store([file_path], vs_path, knowledge_base_id)
    if len(loaded_files) > 0:
        file_status = f"文件 {file_name} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)


async def book_chat(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        llm_answer: bool = Body(True, description="是否启动llm回答", example="true or false"),
        streaming: bool = Body(False, description="流式控制", example="true or false"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
        score_threshold: int = Body(VECTOR_SEARCH_SCORE_THRESHOLD,
                                    description="知识检索内容相关度 Score, 数值范围约为0-1100，"
                                                "如果为0，则不生效，经测试设置为小于500时，匹配结果更精准",
                                    example=0),
        vector_search_top_k: int = Body(VECTOR_SEARCH_TOP_K, description="", example=5),
        chunk_conent: bool = Body(True, description="", example=True),
        chunk_size: int = Body(CHUNK_SIZE, description="匹配后单段上下文长度", example=250),
        max_length: int = Body(2048, description="", example=2048),
        top_p: float = Body(0.70, description="", example=0.70),
        temperature: float = Body(0.10, description="", example=0.10),
):
    query = question
    # vector_store = load_vector_store(vs_path, local_doc_qa.embeddings)
    # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
    vector_store = Milvus(
        embedding_function=local_doc_qa.embeddings,
        collection_name="{}".format(knowledge_base_id),
        connection_args={"host": "127.0.0.1", "port": "19530"},
        search_params={"HNSW": {"metric_type": "L2", "params": {"ef": vector_search_top_k}}},
    )
    vector_store.chunk_conent = chunk_conent
    vector_store.score_threshold = score_threshold
    vector_store.chunk_size = chunk_size

    if llm_answer:  # "知识库问答"
        history = []  # 单轮问答，这样可以避免字符超限制，目前支持最大2048tokens
        for resp, history in local_doc_qa.get_knowledge_based_answer0525(
                vector_store=vector_store,
                my_llm=shared.loaderCheckPoint,
                query=query,
                chat_history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                streaming=streaming):
            pass

        source = "<br> <br>"
        # source += "".join(
        #     [
        #         f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
        #         f"""{doc.page_content}\n"""
        #         f"""</details>"""
        #         for i, doc in
        #         enumerate(resp["source_documents"])])
        source += "".join(
            [
                f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc[0].metadata["source"])[-1]}</summary>\n"""
                f"""{doc[0].page_content}\n"""
                f"""</details>"""
                for i, doc in
                enumerate(resp["source_documents"])])

        history[-1][-1] += source
        result = history[-1][-1]
        # yield history, ""

    else:  # "知识库测试"
        resp, prompt = local_doc_qa.get_knowledge_based_conent_test0614(query=query,
                                                                        chunk_conent=chunk_conent,
                                                                        vector_store=vector_store,
                                                                        score_threshold=score_threshold,
                                                                        vector_search_top_k=vector_search_top_k,
                                                                        chunk_size=chunk_size)
        if not resp["source_documents"]:
            history = []
            # yield history + [[query,
            #                   "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
            result = ''
        else:
            # source = "\n".join(
            #     [
            #         f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
            #         f"""{doc.page_content}\n"""
            #         f"""</details>"""
            #         for i, doc in
            #         enumerate(resp["source_documents"])])
            # history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
            source = "\n".join(
                [
                    f"""<details open> <summary>【知识相关度 Score】：{int(doc[1])} - 【出处{i + 1}】：  {os.path.split(doc[0].metadata["source"])[-1]} </summary>\n"""
                    f"""{doc[0].page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])

            result = history[-1][-1]

            # yield history, ""

    def query_book_info(source_documents_list):
        """
        {
        "id": "1675038829967855631",
        "name": "美国企业年金投资与监管研究",
        "book_isbn": "7562530671",
        "site_isbn": "9787562530671",
        "author": "陈星编",
        "publishing_house": "中国地质大学出版社",
        "publish_date": "2013",
        "doc_page_size": "239",
        "clc_type_id": "F847.126.7",
        "clc_type_name": "经济",
        "pdf_path": "",
        "word_path": "http://114.115.215.96/group3/M00/00/92/wKjIbGSmpsSAAqe9AB4lwEUh0v8008.doc",
        "create_by": "admin",
        "create_time": "1/7/2023 15:09:12",
        "update_by": "",
        "update_time": "6/7/2023 19:34:29",
        "deleted": "0"
        },
        """
        file_name2info = {}
        try:
            for i in source_documents_list:
                doc = i[0]
                file_name = doc.metadata['source'].split('/')[-1]
                with mysql_conn.cursor() as cursor:
                    sql_str = f'''SELECT id, name, book_isbn, site_isbn, author, 
                    publishing_house, publish_date, doc_page_size, clc_type_id, clc_type_name, 
                    pdf_path, word_path FROM ai_report_book WHERE word_path LIKE '%{file_name}' '''
                    print('sql_str={}'.format(sql_str))
                    cursor.execute(sql_str)
                    rows = cursor.fetchall()
                    for row in list(rows):
                        file_name2info[file_name] = {
                            "id": row[0],
                            "name": row[1],
                            "book_isbn": row[2],
                            "site_isbn": row[3],
                            "author": row[4],
                            "publishing_house": row[5],
                            "publish_date": row[6],
                            "page_size": row[7],
                            "clc_type_id": row[8],
                            "clc_type_name": row[9],
                            "pdf_path": row[10],
                            "word_path": row[11]
                        }

                cursor.close()

            return file_name2info
        except Exception as e:
            print('Error: {}'.format(e))
            return file_name2info

    def clean_text(text):
        """
        对结果进行优化。
        """
        text = str(text).strip()
        if str(text).endswith('。'):
            return text
        elif '。' not in text:
            return text
        else:
            text = '。'.join(re.split('[。]', text)[0: -1]) + '。'
            return text

    books_info = query_book_info(resp["source_documents"])

    source_documents = []
    for i, e in enumerate(resp["source_documents"]):
        doc = e[0]
        page_content = clean_text(doc.page_content)
        if len(page_content) <= 1:
            continue

        score = int(e[1])
        current_book = books_info.get(doc.metadata['source'].split('/')[-1], {})

        temp = {
            'page_content': page_content,
            'score': score,
            'from': {
                'chapter': '',
                'book_info': current_book
            }

        }
        source_documents.append(temp)

    source_documents = sorted(source_documents, key=lambda x: x['score'], reverse=False)  # 升序

    resp_json = {
        'question': question,
        'response': result,
        'history': history,
        'source_documents': source_documents
    }
    return JSONResponse(resp_json)


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

    def content2txt0915(id: str, content: str):
        """
        将资讯内容转换为txt 文件，且文件名以id命名
        """
        # 定义文件名，文件名有title的内容组成，但由于文件名需要去除特殊字符，否则编译失败
        file_name = str(id) + ".txt"
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

    file_path, file_name = content2txt0915(id=id, content=content)
    # vs_path 改用miluvs后暂未使用
    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files, _ = local_doc_qa.init_knowledge_vector_store([file_path], vs_path, knowledge_base_id)
    if len(loaded_files) > 0:
        file_status = f"文件 {file_name} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)


async def News_chat(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        llm_answer: bool = Body(True, description="是否启动llm回答", example="true or false"),
        streaming: bool = Body(False, description="流式控制", example="true or false"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
        score_threshold: int = Body(VECTOR_SEARCH_SCORE_THRESHOLD,
                                    description="知识检索内容相关度 Score, 数值范围约为0-1100，"
                                                "如果为0，则不生效，经测试设置为小于500时，匹配结果更精准",
                                    example=0),
        vector_search_top_k: int = Body(VECTOR_SEARCH_TOP_K, description="", example=5),
        chunk_conent: bool = Body(True, description="", example=True),
        chunk_size: int = Body(CHUNK_SIZE, description="匹配后单段上下文长度", example=250),
        max_length: int = Body(2048, description="", example=2048),
        top_p: float = Body(0.70, description="", example=0.70),
        temperature: float = Body(0.10, description="", example=0.10),
):
    query = question
    # vector_store = load_vector_store(vs_path, local_doc_qa.embeddings)
    # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
    vector_store = Milvus(
        embedding_function=local_doc_qa.embeddings,
        collection_name="{}".format(knowledge_base_id),
        connection_args={"host": "127.0.0.1", "port": "19530"},
        search_params={"HNSW": {"metric_type": "L2", "params": {"ef": vector_search_top_k}}},
    )
    vector_store.chunk_conent = chunk_conent
    vector_store.score_threshold = score_threshold
    vector_store.chunk_size = chunk_size

    if llm_answer:  # "知识库问答"
        local_doc_qa.top_k = vector_search_top_k
        history = []  # 单轮问答，这样可以避免字符超限制，目前支持最大2048tokens
        for resp, history in local_doc_qa.get_knowledge_based_answer0525(
                vector_store=vector_store,
                my_llm=shared.loaderCheckPoint,
                query=query,
                chat_history=history,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                streaming=streaming):
            pass

        source = "<br> <br>"
        # source += "".join(
        #     [
        #         f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
        #         f"""{doc.page_content}\n"""
        #         f"""</details>"""
        #         for i, doc in
        #         enumerate(resp["source_documents"])])
        source += "".join(
            [
                f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc[0].metadata["source"])[-1]}</summary>\n"""
                f"""{doc[0].page_content}\n"""
                f"""</details>"""
                for i, doc in
                enumerate(resp["source_documents"])])

        history[-1][-1] += source
        result = history[-1][-1]
        # yield history, ""

    else:  # "知识库测试"
        resp, prompt = local_doc_qa.get_knowledge_based_conent_test0614(query=query,
                                                                        chunk_conent=chunk_conent,
                                                                        vector_store=vector_store,
                                                                        score_threshold=score_threshold,
                                                                        vector_search_top_k=vector_search_top_k,
                                                                        chunk_size=chunk_size)
        if not resp["source_documents"]:
            history = []
            # yield history + [[query,
            #                   "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
            result = ''
        else:
            # source = "\n".join(
            #     [
            #         f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
            #         f"""{doc.page_content}\n"""
            #         f"""</details>"""
            #         for i, doc in
            #         enumerate(resp["source_documents"])])
            # history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
            source = "\n".join(
                [
                    f"""<details open> <summary>【知识相关度 Score】：{int(doc[1])} - 【出处{i + 1}】：  {os.path.split(doc[0].metadata["source"])[-1]} </summary>\n"""
                    f"""{doc[0].page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])

            result = history[-1][-1]

            # yield history, ""

    def query_book_info(source_documents_list):
        """
        {
        "id": "1675038829967855631",
        "name": "美国企业年金投资与监管研究",
        "book_isbn": "7562530671",
        "site_isbn": "9787562530671",
        "author": "陈星编",
        "publishing_house": "中国地质大学出版社",
        "publish_date": "2013",
        "doc_page_size": "239",
        "clc_type_id": "F847.126.7",
        "clc_type_name": "经济",
        "pdf_path": "",
        "word_path": "http://114.115.215.96/group3/M00/00/92/wKjIbGSmpsSAAqe9AB4lwEUh0v8008.doc",
        "create_by": "admin",
        "create_time": "1/7/2023 15:09:12",
        "update_by": "",
        "update_time": "6/7/2023 19:34:29",
        "deleted": "0"
        },
        """
        file_name2info = {}
        try:
            for i in source_documents_list:
                doc = i[0]
                file_name = doc.metadata['source'].split('/')[-1]
                with mysql_conn.cursor() as cursor:
                    sql_str = f'''SELECT id, name, book_isbn, site_isbn, author, 
                    publishing_house, publish_date, doc_page_size, clc_type_id, clc_type_name, 
                    pdf_path, word_path FROM ai_report_book WHERE word_path LIKE '%{file_name}' '''
                    # print('sql_str={}'.format(sql_str))
                    cursor.execute(sql_str)
                    rows = cursor.fetchall()

                    for row in list(rows):
                        file_name2info[file_name] = {
                            "id": row[0],
                            "name": row[1],
                            "book_isbn": row[2],
                            "site_isbn": row[3],
                            "author": row[4],
                            "publishing_house": row[5],
                            "publish_date": row[6],
                            "page_size": row[7],
                            "clc_type_id": row[8],
                            "clc_type_name": row[9],
                            "pdf_path": row[10],
                            "word_path": row[11]
                        }

                cursor.close()

            return file_name2info
        except Exception as e:
            print('Error: {}'.format(e))
            return file_name2info

    def clean_text(text):
        """
        对结果进行优化。
        """
        text = str(text).strip()
        if str(text).endswith('。'):
            return text
        elif '。' not in text:
            return text
        else:
            text = '。'.join(re.split('[。]', text)[0: -1]) + '。'
            return text

    def deduplicate_list(dictionaries, key):
        unique_values = set()
        result = []
        for dictionary in dictionaries:
            value = dictionary[key]
            if value not in unique_values:
                unique_values.add(value)
                result.append(dictionary)
        return result

    # books_info = query_book_info(resp["source_documents"])

    source_documents = []
    file_name_unique = []
    for i, e in enumerate(resp["source_documents"]):
        doc = e[0]
        page_content = clean_text(doc.page_content)
        if len(page_content) <= 2:
            continue

        score = int(e[1])
        # current_book = books_info.get(doc.metadata['source'].split('/')[-1], {})
        file_name = doc.metadata['source'].split('/')[-1]
        if file_name not in file_name_unique:
            file_name_unique.append(file_name)
            file_path = doc.metadata["source"]
            # print(f'当前数据内容为：{doc.metadata["source"]}')
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            temp = {
                'page_content': page_content,
                'file_name': file_name,
                'content': content,
                'score': score,
            }
            source_documents.append(temp)

    # source_documents = deduplicate_list(dictionaries=source_documents, key="file_name")  # 基于file_name 去重
    source_documents = sorted(source_documents, key=lambda x: x['score'], reverse=False)  # 升序

    resp_json = {
        'question': question,
        'response': result,
        'history': history,
        'source_documents': source_documents
    }
    return JSONResponse(resp_json)


async def upload_file(
        file: UploadFile = File(description="A single binary file"),
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file.filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file.filename} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)

    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
    if len(loaded_files) > 0:
        file_status = f"文件 {file.filename} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)


async def upload_files(
        files: Annotated[
            List[UploadFile], File(description="Multiple files as UploadFile")
        ],
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    filelist = []
    for file in files:
        file_content = ''
        file_path = os.path.join(saved_path, file.filename)
        file_content = file.file.read()
        if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
            continue
        with open(file_path, "ab+") as f:
            f.write(file_content)
        filelist.append(file_path)
    if filelist:
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, get_vs_path(knowledge_base_id))
        if len(loaded_files):
            file_status = f"已上传 {'、'.join([os.path.split(i)[-1] for i in loaded_files])} 至知识库，并已加载知识库，请开始提问"
            return BaseResponse(code=200, msg=file_status)
    file_status = "文件未成功加载，请重新上传文件"
    return BaseResponse(code=500, msg=file_status)


async def list_kbs():
    # Get List of Knowledge Base
    if not os.path.exists(KB_ROOT_PATH):
        all_doc_ids = []
    else:
        all_doc_ids = [
            folder
            for folder in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, folder))
               and os.path.exists(os.path.join(KB_ROOT_PATH, folder, "vector_store", "index.faiss"))
        ]

    return ListDocsResponse(data=all_doc_ids)


async def list_docs(
        knowledge_base_id: Optional[str] = Query(default=None, description="Knowledge Base Name", example="kb1")
):
    local_doc_folder = get_folder_path(knowledge_base_id)
    if not os.path.exists(local_doc_folder):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    all_doc_names = [
        doc
        for doc in os.listdir(local_doc_folder)
        if os.path.isfile(os.path.join(local_doc_folder, doc))
    ]
    return ListDocsResponse(data=all_doc_names)


async def delete_kb(
        knowledge_base_id: str = Query(...,
                                       description="Knowledge Base Name",
                                       example="kb1"),
):
    # TODO: 确认是否支持批量删除知识库
    knowledge_base_id = urllib.parse.unquote(knowledge_base_id)
    if not os.path.exists(get_folder_path(knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    shutil.rmtree(get_folder_path(knowledge_base_id))
    return BaseResponse(code=200, msg=f"Knowledge Base {knowledge_base_id} delete success")


async def delete_doc(
        knowledge_base_id: str = Query(...,
                                       description="Knowledge Base Name",
                                       example="kb1"),
        doc_name: str = Query(
            None, description="doc name", example="doc_name_1.pdf"
        ),
):
    knowledge_base_id = urllib.parse.unquote(knowledge_base_id)
    if not os.path.exists(get_folder_path(knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    doc_path = get_file_path(knowledge_base_id, doc_name)
    if os.path.exists(doc_path):
        os.remove(doc_path)
        remain_docs = await list_docs(knowledge_base_id)
        if len(remain_docs.data) == 0:
            shutil.rmtree(get_folder_path(knowledge_base_id), ignore_errors=True)
            return BaseResponse(code=200, msg=f"document {doc_name} delete success")
        else:
            status = local_doc_qa.delete_file_from_vector_store(doc_path, get_vs_path(knowledge_base_id))
            if "success" in status:
                return BaseResponse(code=200, msg=f"document {doc_name} delete success")
            else:
                return BaseResponse(code=1, msg=f"document {doc_name} delete fail")
    else:
        return BaseResponse(code=1, msg=f"document {doc_name} not found")


async def update_doc(
        knowledge_base_id: str = Query(...,
                                       description="知识库名",
                                       example="kb1"),
        old_doc: str = Query(
            None, description="待删除文件名，已存储在知识库中", example="doc_name_1.pdf"
        ),
        new_doc: UploadFile = File(description="待上传文件"),
):
    knowledge_base_id = urllib.parse.unquote(knowledge_base_id)
    if not os.path.exists(get_folder_path(knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    doc_path = get_file_path(knowledge_base_id, old_doc)
    if not os.path.exists(doc_path):
        return BaseResponse(code=1, msg=f"document {old_doc} not found")
    else:
        os.remove(doc_path)
        delete_status = local_doc_qa.delete_file_from_vector_store(doc_path, get_vs_path(knowledge_base_id))
        if "fail" in delete_status:
            return BaseResponse(code=1, msg=f"document {old_doc} delete failed")
        else:
            saved_path = get_folder_path(knowledge_base_id)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)

            file_content = await new_doc.read()  # 读取上传文件的内容

            file_path = os.path.join(saved_path, new_doc.filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
                file_status = f"document {new_doc.filename} already exists"
                return BaseResponse(code=200, msg=file_status)

            with open(file_path, "wb") as f:
                f.write(file_content)

            vs_path = get_vs_path(knowledge_base_id)
            vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
            if len(loaded_files) > 0:
                file_status = f"document {old_doc} delete and document {new_doc.filename} upload success"
                return BaseResponse(code=200, msg=file_status)
            else:
                file_status = f"document {old_doc} success but document {new_doc.filename} upload fail"
                return BaseResponse(code=500, msg=file_status)


async def local_doc_test(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path):
        # return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        return ChatMessage(
            question=question,
            response=f"Knowledge base {knowledge_base_id} not found",
            history=history,
            source_documents=[],
        )
    else:

        resp, prompt = local_doc_qa.get_knowledge_based_conent_test(query=question,
                                                                    vs_path=vs_path,
                                                                    chunk_conent=True)
        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        return ChatMessage(
            question=question,
            response="",
            history=history,
            source_documents=source_documents
        )


async def local_doc_chat(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    vs_path = get_vs_path(knowledge_base_id)
    if not os.path.exists(vs_path):
        # return BaseResponse(code=1, msg=f"Knowledge base {knowledge_base_id} not found")
        return ChatMessage(
            question=question,
            response=f"Knowledge base {knowledge_base_id} not found",
            history=history,
            source_documents=[],
        )
    else:
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            pass
        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        return ChatMessage(
            question=question,
            response=resp["result"],
            history=history,
            source_documents=source_documents,
        )


async def bing_search_chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: Optional[List[List[str]]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    for resp, history in local_doc_qa.get_search_result_based_answer(
            query=question, chat_history=history, streaming=True
    ):
        pass
    source_documents = [
        f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
        for inum, doc in enumerate(resp["source_documents"])
    ]

    return ChatMessage(
        question=question,
        response=resp["result"],
        history=history,
        source_documents=source_documents,
    )


# todo: 0911 增加接口超时处理
#
# async def chat(
#     llm_params: LLM_Params,
#     question: str = Body(..., description="Question", example="工伤保险是什么？"),
#     history: List[List[str]] = Body(
#         [],
#         description="History of previous questions and answers",
#         example=[
#             [
#                 "工伤保险是什么？",
#                 "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
#             ]
#         ],
#     )
# ):
#     try:
#         # 设置超时时间为50秒
#         result = await asyncio.wait_for(
#             chat_with_timeout(llm_params, question, history),
#             timeout=50
#         )
#         return result
#     except asyncio.TimeoutError:
#         return {"error": "Request timed out."}

#
# async def chat_with_timeout(llm_params, question, history):
#     max_length = llm_params.max_length if llm_params.max_length else 2048
#     top_p = llm_params.top_p if llm_params.top_p else 0.70
#     temperature = llm_params.temperature if llm_params.temperature else 0.95
#     print("=" * 4 + f"当前请求内容为: {question[:1000]}" + "=" * 4)
#     print('max_length = {}, top_p = {}, temperature = {}'.format(max_length, top_p, temperature))
#     response, history = model.chat(tokenizer,
#                                    question,
#                                    history,
#                                    max_length=max_length,
#                                    top_p=top_p,
#                                    temperature=temperature)
#
#     # for resp, history in model.stream_chat(tokenizer,
#     #                                        question,
#     #                                        history,
#     #                                        max_length=max_length,
#     #                                        top_p=top_p,
#     #                                        temperature=temperature):
#     #     # pass
#     #     final_resp = resp
#     #     final_history = history
#
#     # todo: 0831 排查显存资源不释放情况
#     torch_gc()
#
#     return ChatMessage(
#         question=question,
#         response=response,
#         history=history,
#         source_documents=[],
#     )
#  todo: 0912 增加异步处理
from fastapi import BackgroundTasks
import torch
# 显存阈值（以字节为单位）
MEMORY_THRESHOLD = 16 * 1024 * 1024 * 1024  # 1GB


async def chat_async(
        llm_params: LLM_Params,
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        )
):
    tokenizer = shared.loaderCheckPoint.tokenizer
    model = shared.loaderCheckPoint.model
    # 异步处理的回调函数
    def process_response(resp, history):
        # 处理响应
        newline_count = 0  # 计数器
        if resp.endswith("\\n"):
            newline_count += 1

        if newline_count >= 5:
            torch_gc()
            return ChatMessage(
                question=question,
                response=resp.strip("\\n\\n\\n\\n\\n"),
                history=history,
                source_documents=[],
            )
        else:
            torch_gc()
            pass

    # 后台任务函数
    def background_task(history):
        for resp, history in model.stream_chat(tokenizer,
                                               question,
                                               history,
                                               max_length=max_length,
                                               top_p=top_p,
                                               temperature=temperature):
            process_response(resp, history)

    # 获取参数
    max_length = llm_params.max_length if llm_params.max_length else 2048
    top_p = llm_params.top_p if llm_params.top_p else 0.70
    temperature = llm_params.temperature if llm_params.temperature else 0.95

    # 启动后台任务
    background_tasks = BackgroundTasks()
    background_tasks.add_task(background_task, history)

    return ChatMessage(
        question=question,
        response="Processing...",
        history=history,
        source_documents=[],
    ), background_tasks


async def chat(
        llm_params: LLM_Params,
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        )
):
    # for answer_result in local_doc_qa.llm.generatorAnswer(prompt=question, history=history,
    #                                                       streaming=True):
    #     resp = answer_result.llm_output["answer"]
    #     history = answer_result.history
    #     pass

    max_length = llm_params.max_length if llm_params.max_length else 2048
    top_p = llm_params.top_p if llm_params.top_p else 0.70
    temperature = llm_params.temperature if llm_params.temperature else 0.95
    # print("="*4 + f"当前请求内容为: {question[:1000]}" + "="*4)
    print('max_length = {}, top_p = {}, temperature = {}'.format(max_length, top_p, temperature))
    newline_count = 0  # 计数器

    resp = ""

    # 监测显存使用
    memory_allocated = torch.cuda.memory_allocated()
    if memory_allocated >= MEMORY_THRESHOLD:
        # 释放显存并重新加载
        torch.cuda.empty_cache()
        # 重新加载模型和Tokenizer等资源
        shared.loaderCheckPoint = LoaderCheckPoint(args_dict)

    for resp, history in model.stream_chat(tokenizer,
                                           question,
                                           history,
                                           max_length=max_length,
                                           top_p=top_p,
                                           temperature=temperature,
                                           ):
        # yield history, ""
        if resp.endswith("\\n"):
            newline_count += 1

        if newline_count >= 5:
            # 达到显存释放条件
            torch.cuda.empty_cache()
            return ChatMessage(
                question=question,
                response=resp.strip("\\n\\n\\n\\n\\n"),
                history=history,
                source_documents=[],
            )
        else:
            pass
    # 达到显存释放条件或完成聊天循环后，再次释放显存
    torch.cuda.empty_cache()

    return ChatMessage(
        question=question,
        response=resp,
        history=history,
        source_documents=[],
    )


async def stream_chat0608(websocket: WebSocket):
    await websocket.accept()
    turn = 1
    while True:
        input_json = await websocket.receive_json()
        question = input_json.get("question", "")
        history = input_json.get("history", [])
        max_length = input_json.get("max_length", 2048)
        top_p = input_json.get("top_p", 0.70)
        temperature = input_json.get("temperature", 0.10)

        # await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        tokenizer = shared.loaderCheckPoint.tokenizer
        model = shared.loaderCheckPoint.model
        last_print_len = 0
        for resp, history in model.stream_chat(tokenizer,
                                               question,
                                               history,
                                               max_length=max_length,
                                               top_p=top_p,
                                               temperature=temperature):
            # print('resp: {}'.format(resp))
            answer = resp[last_print_len:]
            # encoded_answer = answer.encode('utf-8')
            await websocket.send_text(answer)
            last_print_len = len(resp)

        torch_gc()

        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                },
                ensure_ascii=False,
            )
        )
        turn += 1


async def stream_chat(websocket: WebSocket, knowledge_base_id: str):
    await websocket.accept()
    turn = 1
    while True:
        input_json = await websocket.receive_json()
        question, history, knowledge_base_id = input_json["question"], input_json["history"], input_json[
            "knowledge_base_id"]
        vs_path = get_vs_path(knowledge_base_id)

        if not os.path.exists(vs_path):
            await websocket.send_json({"error": f"Knowledge base {knowledge_base_id} not found"})
            await websocket.close()
            return

        await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            await websocket.send_text(resp["result"][last_print_len:])
            last_print_len = len(resp["result"])

        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                    "sources_documents": source_documents,
                },
                ensure_ascii=False,
            )
        )
        turn += 1


async def text_summary(
        llm_params: LLM_Params,
        content: str = Body(..., description="Content"),
        summary_max_length: int = Body(..., description="摘要的最大长度", example=256),
):
    prompt = prompt_templete.prompt_templete4summary.replace('{context}', content).replace('{summary_max_length}',
                                                                                           str(summary_max_length))
    try:
        history = []
        tokenizer = shared.loaderCheckPoint.tokenizer
        model = shared.loaderCheckPoint.model
        response, history = model.chat(tokenizer,
                                       prompt,
                                       history=history,
                                       max_length=llm_params.max_length if llm_params.max_length else 2048,
                                       top_p=llm_params.top_p if llm_params.top_p else 0.7,
                                       temperature=llm_params.temperature if llm_params.temperature else 0.95)

        text = response
        if len(text) > summary_max_length:
            # print("正文长度超过最大长度，已截取前{}个字符：".format(summary_max_length))
            sentences = text.split('。')
            result = ""
            for s in sentences:
                if len(result) + len(s) + 1 <= summary_max_length:
                    result += s + "。"
                else:
                    break
            # print(result)
            text = result
        else:
            pass
            # print("正文长度未超过最大长度：")
            # print(text)

        return SummaryResponse(
            code=200,
            msg="success",
            summary=text
        )

    except Exception as e:

        return SummaryResponse(
            code=201,
            msg="fail. Error: {}".format(e),
            summary=""
        )


async def stream_chat(websocket: WebSocket, knowledge_base_id: str):
    await websocket.accept()
    turn = 1
    while True:
        input_json = await websocket.receive_json()
        question, history, knowledge_base_id = input_json["question"], input_json["history"], input_json[
            "knowledge_base_id"]
        vs_path = get_vs_path(knowledge_base_id)

        if not os.path.exists(vs_path):
            await websocket.send_json({"error": f"Knowledge base {knowledge_base_id} not found"})
            await websocket.close()
            return

        await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            await asyncio.sleep(0)
            await websocket.send_text(resp["result"][last_print_len:])
            last_print_len = len(resp["result"])

        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                    "sources_documents": source_documents,
                },
                ensure_ascii=False,
            )
        )
        turn += 1


async def document():
    return RedirectResponse(url="/docs")


print('#########1')
# parser.add_argument("--host", type=str, default="0.0.0.0")
# parser.add_argument("--port", type=int, default=7861)
print('#########2')
# 初始化消息
print('###########3')
# args = None
# args = parser.parse_args()
# print('###########4')
# args_dict = vars(args)
# print(args_dict)

args_dict = {'no_remote_model': False, 'model_name': 'chatglm2-6b-32k', 'use_ptuning_v2': True, 'lora': None, 'lora_dir': 'loras/',
             'load_in_8bit': False, 'bf16': False, 'workers': None, 'host': '0.0.0.0', 'port': 7861}

app = FastAPI()

# def api_start(host="0.0.0.0", port=7861, my_shared=None, my_local_doc_qa=None):
# global app
global local_doc_qa
global shared
# global model
# global tokenizer

# if my_shared:
#     shared = my_shared
#
# if my_local_doc_qa:
#     local_doc_qa = my_local_doc_qa

shared.loaderCheckPoint = LoaderCheckPoint(args_dict)

llm_model_ins = shared.loaderLLM(use_ptuning_v2=args_dict.get('use_ptuning_v2', False))
llm_model_ins.set_history_len(LLM_HISTORY_LEN)

tokenizer = shared.loaderCheckPoint.tokenizer
model = shared.loaderCheckPoint.model

local_doc_qa = LocalDocQA()
local_doc_qa.init_cfg(
    llm_model=llm_model_ins,
    embedding_model=EMBEDDING_MODEL,
    embedding_device=EMBEDDING_DEVICE,
    top_k=VECTOR_SEARCH_TOP_K,
)


# llm_model_ins = shared.loaderLLM()
# llm_model_ins.set_history_len(LLM_HISTORY_LEN)

# app = FastAPI()
# Add CORS middleware to allow all origins
# 在config.py中设置OPEN_DOMAIN=True，允许跨域
# set OPEN_DOMAIN=True in config.py to allow cross-domain
if OPEN_CROSS_DOMAIN:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# # app.websocket("/local_doc_qa/stream-chat/{knowledge_base_id}")(stream_chat)
app.get("/", response_model=BaseResponse)(document)
app.post("/local_doc_qa/chat", response_model=ChatMessage)(chat)
app.websocket("/local_doc_qa/stream_chat")(stream_chat0608)
#
app.post("/local_doc_qa/upload_file", response_model=BaseResponse)(upload_file)
app.post("/local_doc_qa/upload_files", response_model=BaseResponse)(upload_files)
app.post("/local_doc_qa/local_doc_chat", response_model=ChatMessage)(local_doc_chat)
app.post("/local_doc_qa/bing_search_chat", response_model=ChatMessage)(bing_search_chat)
app.get("/local_doc_qa/list_knowledge_base", response_model=ListDocsResponse)(list_kbs)
app.get("/local_doc_qa/list_files", response_model=ListDocsResponse)(list_docs)
app.delete("/local_doc_qa/delete_knowledge_base", response_model=BaseResponse)(delete_kb)
app.delete("/local_doc_qa/delete_file", response_model=BaseResponse)(delete_doc)
app.post("/local_doc_qa/update_file", response_model=BaseResponse)(update_doc)

# book manage api
app.post("/book/upload_book", response_model=BaseResponse)(upload_book)
app.post("/book/book_chat", response_model=ChatMessage)(book_chat)

# News manage api
app.post("/news/upload_news", response_model=BaseResponse)(upload_News)
app.post("/news/news_chat", response_model=ChatMessage)(News_chat)

# 线索追踪分析系统 api
app.post("/xiansuo/upload_file", response_model=BaseResponse)(xiansuo_upload_file)
app.websocket("/xiansuo/stream_chat")(xiansuo_stream_chat)

#
# # 合同 api
# app.post("/contract/extract_contract_info", response_model=BaseResponse)(extract_contract_info)

# local_doc_qa = LocalDocQA()
# local_doc_qa.init_cfg(
#     llm_model=llm_model_ins,
#     embedding_model=EMBEDDING_MODEL,
#     embedding_device=EMBEDDING_DEVICE,
#     top_k=VECTOR_SEARCH_TOP_K,
# )

# uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861)
