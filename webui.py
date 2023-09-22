import gradio as gr
import shutil

from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import os
from chains.local_doc_qa import load_vector_store
from utils import torch_gc
from langchain.vectorstores import Milvus

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst


embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()

flag_csv_logger = gr.CSVLogger()


def get_answer(query, vs_path, history, mode, vector_store,
               score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
               vector_search_top_k=VECTOR_SEARCH_TOP_K,
               chunk_conent: bool = True,
               chunk_size=CHUNK_SIZE,
               streaming: bool = STREAMING,
               max_length: int = 30000,
               top_p: float = 0.70,
               temperature: float = 0.10,
               ):
    # params = {
    #     'query': query,
    #     'vs_path': vs_path,
    #     'history': history,
    #     'mode': mode,
    #     'score_threshold': score_threshold,
    #     'vector_search_top_k': vector_search_top_k,
    #     'chunk_conent': chunk_conent,
    #     'chunk_size': chunk_size,
    #     'streaming': streaming,
    #     'max_length': max_length,
    #     'top_p': top_p,
    #     'temperature': temperature,
    # }
    # logger.info(json.dumps(params, ensure_ascii=False, indent=2))

    if mode == "Bing搜索问答":
        for resp, history in local_doc_qa.get_search_result_based_answer(
                query=query, chat_history=history, streaming=streaming):
            source = "\n\n"
            source += "".join(
                [
                    f"""
                    <details> <summary style="color:blue">出处 [{i + 1}] <a href="{doc.metadata["source"]}" target="_blank">{doc.metadata["source"]}</a> </summary>\n"""
                    f"""{doc.page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            history[-1][-1] += source
            yield history, ""
    # elif mode == "知识库问答":
    #     print('vs_path: {}'.format(vs_path))
    #     knowledge_base = os.path.split(os.path.split(vs_path)[0])[-1]
    #     print('knowledge_base: {}'.format(knowledge_base))
    #     try:
    #         vector_store = Milvus(
    #             embedding_function=local_doc_qa.embeddings,
    #             collection_name='{}'.format(knowledge_base),
    #             connection_args={"host": "127.0.0.1", "port": "19530"},
    #             search_params={"HNSW": {"metric_type": "L2", "params": {"ef": vector_search_top_k}}},
    #         )
    #     except Exception as e:
    #         print('Error: {}'.format(e))
    #
    #     history = []  # 单轮问答，这样可以避免字符超限制，目前支持最大2048tokens
    #     if vs_path is not None and os.path.exists(vs_path) and vector_store is not None:
    #         for resp, history in local_doc_qa.get_knowledge_based_answer0525(
    #                 vector_store=vector_store,
    #                 my_llm=shared.loaderCheckPoint,
    #                 query=query,
    #                 chat_history=history,
    #                 max_length=max_length,
    #                 top_p=top_p,
    #                 temperature=temperature,
    #                 streaming=streaming):
    #             # source = "<br> <br>"
    #             # source += "".join(
    #             #     [
    #             #         f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
    #             #         f"""{doc.page_content}\n"""
    #             #         f"""</details>"""
    #             #         for i, doc in
    #             #         enumerate(resp["source_documents"])])
    #
    #             # history[-1][-1] += source
    #             yield history, ""
    #
    #         source = "<br> <br>"
    #         source += "".join(
    #             [
    #                 f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc[0].metadata["source"])[-1]}</summary>\n"""
    #                 f"""{doc[0].page_content}\n"""
    #                 f"""</details>"""
    #                 for i, doc in
    #                 enumerate(resp["source_documents"])])
    #
    #         history[-1][-1] += source
    #         yield history, ""
    #
    #     else:
    #         input = query
    #         tokenizer = shared.loaderCheckPoint.tokenizer
    #         for response, history in shared.loaderCheckPoint.model.stream_chat(tokenizer,
    #                                                                            input, history,
    #                                                                            max_length=max_length,
    #                                                                            top_p=top_p,
    #                                                                            temperature=temperature):
    #             history = [list(i) for i in history]
    #             history[-1][-1] = response + (
    #                 "\n\n⚠️当前知识库为空，如需基于知识库进行问答，请先加载知识库后，再进行提问。")
    #
    #             yield history, ""

    elif mode == "知识库问答":
        print('vs_path: {}'.format(vs_path))
        knowledge_base = os.path.split(os.path.split(vs_path)[0])[-1]
        print('knowledge_base: {}'.format(knowledge_base))
        try:
            vector_store = Milvus(
                embedding_function=local_doc_qa.embeddings,
                collection_name='{}'.format(knowledge_base),
                connection_args={"host": "127.0.0.1", "port": "19530"},
                # 暂时修改为固定的值 10
                # search_params={"HNSW": {"metric_type": "L2", "params": {"ef": 10}}},
                search_params={"HNSW": {"metric_type": "L2", "params": {"ef": vector_search_top_k}}},
            )
        except Exception as e:
            print('Error: {}'.format(e))

        history = []  # 单轮问答，这样可以避免字符超限制，目前最大30000tokens
        if vs_path is not None and os.path.exists(vs_path) and vector_store is not None:
            for resp, history in local_doc_qa.get_knowledge_based_answer0830(
                    vector_store=vector_store,
                    my_llm=shared.loaderCheckPoint,
                    query=query,
                    chat_history=history,
                    max_length=max_length,
                    top_p=top_p,
                    temperature=temperature,
                    streaming=streaming):
                # source = "<br> <br>"
                # source += "".join(
                #     [
                #         f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
                #         f"""{doc.page_content}\n"""
                #         f"""</details>"""
                #         for i, doc in
                #         enumerate(resp["source_documents"])])

                # history[-1][-1] += source
                yield history, ""

            source = "<br> <br>"
            source += "".join(
                [
                    f"""<details> <summary style="color:blue">出处 [{i + 1}] {os.path.split(doc[0].metadata["source"])[-1]}</summary>\n"""
                    f"""{doc[0].page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])

            history[-1][-1] += source
            yield history, ""

        else:
            input = query
            tokenizer = shared.loaderCheckPoint.tokenizer
            for response, history in shared.loaderCheckPoint.model.stream_chat(tokenizer,
                                                                               input, history,
                                                                               max_length=max_length,
                                                                               top_p=top_p,
                                                                               temperature=temperature):
                history = [list(i) for i in history]
                history[-1][-1] = response + (
                    "\n\n⚠️当前知识库为空，如需基于知识库进行问答，请先加载知识库后，再进行提问。")

                yield history, ""

    elif mode == "知识库测试":
        if vs_path is not None and os.path.exists(vs_path) and vector_store is not None:
            resp, prompt = local_doc_qa.get_knowledge_based_conent_test0614(query=query,
                                                                            vs_path=vs_path,
                                                                            chunk_conent=chunk_conent,
                                                                            vector_store=vector_store,
                                                                            score_threshold=score_threshold,
                                                                            vector_search_top_k=vector_search_top_k,
                                                                            chunk_size=chunk_size)
            if not resp["source_documents"]:
                yield history + [[query,
                                  "根据您的设定，没有匹配到任何内容，请确认您设置的知识相关度 Score 阈值是否过小或其他参数是否正确。"]], ""
            else:
                source = "\n".join(
                    [
                        f"""<details open> <summary>【知识相关度 Score】：{doc.metadata["score"]} - 【出处{i + 1}】：  {os.path.split(doc.metadata["source"])[-1]} </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>"""
                        for i, doc in
                        enumerate(resp["source_documents"])])
                history.append([query, "以下内容为知识库中满足设置条件的匹配结果：\n\n" + source])
                yield history, ""
        else:
            yield history + [[query,
                              "请选择知识库后进行测试，当前未选择知识库。"]], ""

    elif mode == "LLM 对话":
        input = query
        tokenizer = shared.loaderCheckPoint.tokenizer
        for response, history in shared.loaderCheckPoint.model.stream_chat(tokenizer,
                                                                           input,
                                                                           history,
                                                                           max_length=max_length,
                                                                           max_new_tokens=30000,
                                                                           top_p=top_p,
                                                                           temperature=temperature):
            yield history, ""

    else:
        for answer_result in local_doc_qa.llm.generatorAnswer(prompt=query, history=history,
                                                              streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][-1] = resp
            yield history, ""

    torch_gc()

    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},vs_path={vs_path},mode={mode},history={history}")
    flag_csv_logger.flag([query, vs_path, history, mode], username=FLAG_USER_NAME)


# 初始化模型
def init_model():
    # todo: 加载自定义参数
    args_dict = {'no_remote_model': False, 'model_name': 'chatglm2-6b-32k', 'use_ptuning_v2': True, 'lora': None,
                 'lora_dir': 'loras/',
                 'load_in_8bit': False, 'bf16': False}

    shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    llm_model_ins = shared.loaderLLM(use_ptuning_v2=args_dict.get('use_ptuning_v2', False),
                                     )
    llm_model_ins.set_history_len(LLM_HISTORY_LEN)
    try:
        local_doc_qa.init_cfg(llm_model=llm_model_ins)
        generator = local_doc_qa.llm.generatorAnswer("你好")
        for answer_result in generator:
            print(answer_result.llm_output)
        reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
                        " https://github.com/imClumsyPanda/langchain-ChatGLM")
        else:
            logger.info(reply)
        return reply


# def init_model():
#     args = parser.parse_args()
#
#     args_dict = vars(args)
#     shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
#     llm_model_ins = shared.loaderLLM()
#     llm_model_ins.set_history_len(LLM_HISTORY_LEN)
#     try:
#         local_doc_qa.init_cfg(llm_model=llm_model_ins)
#         generator = local_doc_qa.llm.generatorAnswer("你好")
#         for answer_result in generator:
#             print(answer_result.llm_output)
#         reply = """模型已成功加载，可以开始对话，或从右侧选择模式后开始对话"""
#         logger.info(reply)
#         return reply
#     except Exception as e:
#         logger.error(e)
#         reply = """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
#         if str(e) == "Unknown platform: darwin":
#             logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题："
#                         " https://github.com/imClumsyPanda/langchain-ChatGLM")
#         else:
#             logger.info(reply)
#         return reply


def reinit_model(llm_model, embedding_model, llm_history_len, no_remote_model, use_ptuning_v2, use_lora, top_k,
                 history):
    logger.info('当前选中模型是：{}'.format(llm_model))
    try:
        llm_model_ins = shared.loaderLLM(llm_model, no_remote_model, use_ptuning_v2)
        llm_model_ins.history_len = llm_history_len
        local_doc_qa.init_cfg(llm_model=llm_model_ins,
                              embedding_model=embedding_model,
                              top_k=top_k)
        model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
        logger.info(model_status)
    except Exception as e:
        logger.error(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
        logger.info(model_status)
    return history + [[None, model_status]]


def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    # todo: bug   存图片的.pdf文件处理不了。上传失败！ 2023-07-15
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    filelist = []
    if local_doc_qa.llm and local_doc_qa.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
                filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
            vs_path, loaded_files, vector_store = local_doc_qa.init_knowledge_vector_store(filelist,
                                                                                           vs_path,
                                                                                           vs_id,
                                                                                           sentence_size)
        else:
            vs_path, loaded_files, vector_store = local_doc_qa.one_knowledge_add(vs_path, files, one_conent,
                                                                                 one_content_segmentation,
                                                                                 sentence_size)

        vector_store.chunk_size = local_doc_qa.chunk_size
        vector_store.chunk_conent = local_doc_qa.chunk_conent
        vector_store.score_threshold = local_doc_qa.score_threshold

        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
        vector_store = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]], vector_store, \
           gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path) if vs_path else [])


def reset_state():
    return [], []


def stop_generate():
    return True


def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history, \
               gr.update(choices=[]), gr.update(visible=False)
    else:
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        if "index.faiss" in os.listdir(vs_path):
            logger.info("vector is loading...")
            vector_store = load_vector_store(vs_path, local_doc_qa.embeddings)
            logger.info("load vector is complete !")
            vector_store.chunk_size = local_doc_qa.chunk_size
            vector_store.chunk_conent = local_doc_qa.chunk_conent
            vector_store.score_threshold = local_doc_qa.score_threshold
            # print('type(vector_store) = {}'.format(type(vector_store)))
            logger.info(f"已加载知识库{vs_id}，请开始提问")
            file_status = f"已加载知识库{vs_id}，请开始提问"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   vs_path, history + [[None, file_status]], vector_store, \
                   gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), \
                   gr.update(visible=True)
        else:
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请先上传文件后，再开始提问"
            vector_store = None

        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
               vs_path, history + [[None, file_status]], vector_store, \
               gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), \
               gr.update(visible=True)


knowledge_base_test_mode_info = ("【注意】\n\n"
                                 "1. 您已进入知识库测试模式，您输入的任何对话内容都将用于进行知识库查询，"
                                 "并仅输出知识库匹配出的内容及相似度分值和及输入的文本源路径，查询的内容并不会进入模型查询。\n\n"
                                 "2. 知识相关度 Score 经测试，建议设置为 500 或更低，具体设置情况请结合实际使用调整。"
                                 """3. 使用"添加单条数据"添加文本至知识库时，内容如未分段，则内容越多越会稀释各查询内容与之关联的score阈值。\n\n"""
                                 "4. 单条内容长度建议设置在100-150左右。\n\n"
                                 "5. 本界面用于知识入库及知识匹配相关参数设定，但当前版本中，"
                                 "本界面中修改的参数并不会直接修改对话界面中参数，仍需前往`configs/model_config.py`修改后生效。"
                                 "相关参数将在后续版本中支持本界面直接修改。")


def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), history
        # + [[None, "【注意】：您已进入知识库问答模式，您输入的任何查询都将进行知识库查询，然后会自动整理知识库关联内容进入模型查询！！！"]]
    elif mode == "知识库测试":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), [[None,
                                                                                             knowledge_base_test_mode_info]]
    elif mode == "LLM 对话":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), history
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), history


def change_chunk_conent(mode, label_conent, history):
    conent = ""
    if "chunk_conent" in label_conent:
        conent = "搜索结果上下文关联"
    elif "one_content_segmentation" in label_conent:  # 这里没用上，可以先留着
        conent = "内容分段入库"

    if mode:
        return gr.update(visible=True), history + [[None, f"【已开启{conent}】"]]
    else:
        return gr.update(visible=False), history + [[None, f"【已关闭{conent}】"]]


def add_vs_name(vs_name, chatbot):
    if vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "content")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "content"))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "vector_store")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "vector_store"))
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_name), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot, gr.update(visible=True)


# 自动化加载固定文件间中文件
def reinit_vector_store(vs_id, history):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id, "vector_store"))
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                  label="文本入库分句长度限制",
                                  interactive=True, visible=True)
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(os.path.join(KB_ROOT_PATH, vs_id, "content"),
                                                                         vs_path, sentence_size)
        model_status = """知识库构建成功"""
    except Exception as e:
        logger.error(e)
        model_status = """知识库构建未成功"""
        logger.info(model_status)
    return history + [[None, model_status]]


def refresh_vs_list():
    return gr.update(choices=get_vs_list()), gr.update(choices=get_vs_list())


def verify_password(password):
    correct_password = "zzsn9988"  # 替换为实际密码
    if password == correct_password:
        message = "密码验证成功！现在执行特定操作。"
        print(message)
        return gr.update(visible=False), gr.update(visible=True)
    else:
        message = "密码错误，请重试。"
        print(message)
        return gr.update(visible=True), gr.update(visible=False)


def delete_file(vs_id, files_to_delete, chatbot):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    content_path = os.path.join(KB_ROOT_PATH, vs_id, "content")
    docs_path = [os.path.join(content_path, file) for file in files_to_delete]
    status = local_doc_qa.delete_file_from_vector_store(vs_path=vs_path,
                                                        filepath=docs_path)
    if "fail" not in status:
        for doc_path in docs_path:
            if os.path.exists(doc_path):
                os.remove(doc_path)
    rested_files = local_doc_qa.list_file_from_vector_store(vs_path)
    if "fail" in status:
        vs_status = "文件删除失败。"
    elif len(rested_files) > 0:
        vs_status = "文件删除成功。"
    else:
        vs_status = f"文件删除成功，知识库{vs_id}中无已上传文件，请先上传文件后，再开始提问。"
    logger.info(",".join(files_to_delete) + vs_status)
    chatbot = chatbot + [[None, vs_status]]
    return gr.update(choices=local_doc_qa.list_file_from_vector_store(vs_path), value=[]), chatbot


def delete_vs(vs_id, chatbot):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id))
        status = f"成功删除知识库{vs_id}"
        logger.info(status)
        chatbot = chatbot + [[None, status]]
        return gr.update(choices=get_vs_list(), value=get_vs_list()[0]), gr.update(visible=True), gr.update(
            visible=True), \
               gr.update(visible=False), chatbot, gr.update(visible=False)
    except Exception as e:
        logger.error(e)
        status = f"删除知识库{vs_id}失败"
        chatbot = chatbot + [[None, status]]
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), \
               gr.update(visible=True), chatbot, gr.update(visible=True)


block_css = """.importantButton {
        background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
        border: none !important;
    }
    .importantButton:hover {
        background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
        border: none !important;
    }"""

webui_title = """
# 🎉CiglobalGPT-6B🎉
"""
default_vs = get_vs_list()[0] if len(get_vs_list()) > 1 else "为空"
init_message = f"""欢迎使用 langchain-ChatGLM Web UI！

请在右侧切换模式，目前支持直接与 LLM 模型对话或基于本地知识库问答。

知识库问答模式，选择知识库名称后，即可开始问答，当前知识库{default_vs}，如有需要可以在选择知识库名称后上传文件/文件夹至知识库。

知识库暂不支持文件删除，该功能将在后续版本中推出。
"""

# 初始化消息
model_status = init_model()


def start_api():
    import api
    import threading
    t = threading.Thread(target=api.api_start, args=('0.0.0.0', 7861, shared, local_doc_qa,), daemon=True)
    t.start()
    logger.info('api 开启成功！')


# # 启动api
# start_api()

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    vs_path, file_status, model_status, vector_store, score_threshold, vector_search_top_k, chunk_conent, chunk_sizes, \
    streaming = \
        gr.State(os.path.join(KB_ROOT_PATH, get_vs_list()[0]) if len(get_vs_list()) > 1 else ""), \
        gr.State(""), \
        gr.State(model_status), \
        gr.State(None), \
        gr.Number(VECTOR_SEARCH_SCORE_THRESHOLD, visible=False), \
        gr.Number(VECTOR_SEARCH_TOP_K, visible=False), \
        gr.Number(CHUNK_SIZE, visible=False), \
        gr.State(True), \
        gr.State(STREAMING)

    gr.Markdown(webui_title)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)

                with gr.Row():
                    query = gr.Textbox(show_label=False,
                                       placeholder="请输入提问内容，按回车进行提交").style(container=False)
                    send_submit = gr.Button("发送🚀").style(full_width=False, size='lg')

            with gr.Column(scale=5):
                # mode = gr.Radio(["LLM 对话", "知识库问答", "Bing搜索问答"],
                #                 label="请选择使用模式",
                #                 value="知识库问答", )
                mode = gr.Radio(["LLM 对话", "知识库问答"],
                                label="请选择使用模式",
                                value="知识库问答", )
                # mode = gr.Radio(["知识库问答",],
                #                 label="请选择使用模式",
                #                 value="知识库问答", )
                knowledge_set = gr.Accordion("知识库设定", visible=False)
                vs_setting = gr.Accordion("配置知识库")
                chat_session_setting = gr.Accordion("对话参数配置", open=False, visible=False)
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chat_session_setting, chatbot])
                with chat_session_setting:
                    with gr.Column(scale=1):
                        args = parser.parse_args()
                        args_dict = vars(args)
                        logger.info(args_dict)
                        model_name = args_dict.get('model_name')
                        if model_name in ['chatglm-6b']:
                            max_length_default = 2048
                        elif model_name in ['chatglm2-6b']:
                            max_length_default = 8192
                        else:
                            max_length_default = 2048

                        max_length = gr.Slider(0, max_length_default, value=max_length_default, step=1.0,
                                               label="Maximum length",
                                               interactive=True)
                        top_p = gr.Slider(0, 1, value=0.70, step=0.01, label="Top P", interactive=True)
                        temperature = gr.Slider(0, 1, value=0.10, step=0.01, label="Temperature", interactive=True)

                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
                                            )
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    vs_delete = gr.Button("删除本知识库", visible=False)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf', '.png', '.jpg', ".csv"],
                                            file_count="multiple",
                                            show_label=False)
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("删除文件"):
                            files_to_delete = gr.CheckboxGroup(choices=[],
                                                               label="请从知识库已有文件中选择要删除的文件",
                                                               interactive=True)
                            delete_file_button = gr.Button("从知识库中删除选中文件")
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, chatbot],
                                 outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                    vs_delete.click(fn=delete_vs,
                                    inputs=[select_vs, chatbot],
                                    outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                    select_vs.change(fn=change_vs_name_input,
                                     inputs=[select_vs, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot, vector_store, files_to_delete,
                                              vs_delete])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot, vector_store, files_to_delete], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot, vector_store, files_to_delete], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    history = gr.State([])

                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode, vector_store,
                                  score_threshold,
                                  vector_search_top_k,
                                  chunk_conent,
                                  chunk_sizes,
                                  streaming,
                                  max_length, top_p, temperature],
                                 [chatbot, query])

                    send_submit.click(get_answer,
                                      [query, vs_path, chatbot, mode, vector_store,
                                       score_threshold,
                                       vector_search_top_k,
                                       chunk_conent,
                                       chunk_sizes,
                                       streaming,
                                       max_length, top_p, temperature],
                                      [chatbot, query])

                    delete_file_button.click(delete_file,
                                             show_progress=True,
                                             inputs=[select_vs, files_to_delete, chatbot],
                                             outputs=[files_to_delete, chatbot])

                with gr.Row():
                    # stopBtn4ChatSession = gr.Button(value="停止生成", variant='stop')
                    emptyBtn4ChatSession = gr.Button(value="清空对话", variant='primary')

        # stopBtn4ChatSession.click(stop_generate, show_progress=True)
        emptyBtn4ChatSession.click(reset_state, outputs=[chatbot, history], show_progress=True)

    with gr.Tab("知识库测试 Beta"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, knowledge_base_test_mode_info]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)

                with gr.Row():
                    query = gr.Textbox(show_label=False,
                                       placeholder="请输入提问内容，按回车进行提交").style(container=False)
                    send_submit = gr.Button("发送🚀").style(full_width=False, size='lg')

            with gr.Column(scale=5):
                mode = gr.Radio(["知识库测试"],  # "知识库问答",
                                label="请选择使用模式",
                                value="知识库测试",
                                visible=False)
                knowledge_set = gr.Accordion("知识库设定", visible=True)
                vs_setting = gr.Accordion("配置知识库", visible=True)
                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[vs_setting, knowledge_set, chatbot])
                with knowledge_set:
                    score_threshold = gr.Number(value=VECTOR_SEARCH_SCORE_THRESHOLD,
                                                label="知识相关度 Score 阈值，分值越低匹配度越高",
                                                precision=0,
                                                interactive=True)
                    vector_search_top_k = gr.Number(value=VECTOR_SEARCH_TOP_K, precision=0,
                                                    label="获取知识库内容条数", interactive=True)
                    chunk_conent = gr.Checkbox(value=False,
                                               label="是否启用上下文关联",
                                               interactive=True)
                    chunk_sizes = gr.Number(value=CHUNK_SIZE, precision=0,
                                            label="匹配单段内容的连接上下文后最大长度",
                                            interactive=True, visible=False)
                    chunk_conent.change(fn=change_chunk_conent,
                                        inputs=[chunk_conent, gr.Textbox(value="chunk_conent", visible=False),
                                                chatbot],
                                        outputs=[chunk_sizes, chatbot])
                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs_test = gr.Dropdown(get_vs_list(),
                                                 label="请选择要加载的知识库",
                                                 interactive=True,
                                                 value=get_vs_list()[0] if len(get_vs_list()) > 0 else None)
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加单条内容或文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf'],
                                            file_count="multiple",
                                            show_label=False
                                            )
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("添加单条内容"):
                            one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题", lines=1)
                            one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容", lines=5)
                            one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
                                                                   interactive=True)
                            load_conent_button = gr.Button("添加内容并加载知识库")
                    # 将上传的文件保存到content文件夹下,并更新下拉框
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs_test)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, chatbot],
                                 outputs=[select_vs_test, vs_name, vs_add, file2vs, chatbot])
                    select_vs_test.change(fn=change_vs_name_input,
                                          inputs=[select_vs_test, chatbot],
                                          outputs=[vs_name, vs_add, file2vs, vs_path, chatbot, vector_store])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs_test, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot, vector_store], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs_test, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot, vector_store], )
                    load_conent_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs_test, one_title, sentence_size, chatbot,
                                                     one_conent, one_content_segmentation],
                                             outputs=[vs_path, files, chatbot, vector_store], )
                    flag_csv_logger.setup([query, vs_path, chatbot, mode], "flagged")
                    query.submit(get_answer,
                                 [query, vs_path, chatbot, mode, vector_store,
                                  score_threshold,
                                  vector_search_top_k,
                                  chunk_conent,
                                  chunk_sizes,
                                  streaming],
                                 [chatbot, query])
                    send_submit.click(get_answer,
                                      [query, vs_path, chatbot, mode, vector_store,
                                       score_threshold,
                                       vector_search_top_k,
                                       chunk_conent,
                                       chunk_sizes,
                                       streaming],
                                      [chatbot, query])
                with gr.Row():
                    # stopBtn4ChatSession = gr.Button(value="停止生成", variant='stop')
                    emptyBtn4ChatSession = gr.Button(value="清空对话", variant='primary')
        emptyBtn4ChatSession.click(reset_state, outputs=[chatbot, history], show_progress=True)

    with gr.Tab("模型配置"):
        model_config_ui = gr.Accordion("模型配置参数", visible=False)
        passwd_author_ui = gr.Accordion("密码认证", visible=True)
        with passwd_author_ui:
            password = gr.Textbox(label="管理员密码", placeholder="请输入管理员密码", lines=1)
            passwd_sumbit_button = gr.Button("提交")
            passwd_sumbit_button.click(verify_password, [password], [passwd_author_ui, model_config_ui])

        with model_config_ui:
            llm_model = gr.Radio(llm_model_dict_list,
                                 label="LLM 模型",
                                 value=LLM_MODEL,
                                 interactive=True)
            no_remote_model = gr.Checkbox(shared.LoaderCheckPoint.no_remote_model,
                                          label="加载本地模型",
                                          interactive=True)

            llm_history_len = gr.Slider(0, 10,
                                        value=LLM_HISTORY_LEN,
                                        step=1,
                                        label="LLM 对话轮数",
                                        interactive=True)
            use_ptuning_v2 = gr.Checkbox(USE_PTUNING_V2,
                                         label="使用p-tuning-v2微调过的模型",
                                         interactive=True)
            use_lora = gr.Checkbox(USE_LORA,
                                   label="使用lora微调的权重",
                                   interactive=True)
            embedding_model = gr.Radio(embedding_model_dict_list,
                                       label="Embedding 模型",
                                       value=EMBEDDING_MODEL,
                                       interactive=True)
            top_k = gr.Slider(1, 20, value=VECTOR_SEARCH_TOP_K, step=1,
                              label="向量匹配 top k", interactive=True)
            load_model_button = gr.Button("重新加载模型")
            load_model_button.click(reinit_model, show_progress=True,
                                    inputs=[llm_model, embedding_model, llm_history_len, no_remote_model,
                                            use_ptuning_v2,
                                            use_lora, top_k, chatbot], outputs=chatbot)
            load_knowlege_button = gr.Button("重新构建知识库")
            load_knowlege_button.click(reinit_vector_store, show_progress=True,
                                       inputs=[select_vs, chatbot], outputs=chatbot)
    demo.load(
        fn=refresh_vs_list,
        inputs=None,
        outputs=[select_vs, select_vs_test],
        queue=True,
        show_progress=False,
    )

(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=7860,
         show_api=False,
         share=False,
         inbrowser=False))
