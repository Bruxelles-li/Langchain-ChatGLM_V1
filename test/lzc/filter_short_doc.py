import os
import re
import sys

sys.path.append("../")

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from configs.model_config import *

from langchain.text_splitter import CharacterTextSplitter
from typing import List
from utils import torch_gc
from tqdm import tqdm
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from langchain.document_loaders import Docx2txtLoader
import numpy as np
from textsplitter.zh_title_enhance import zh_title_enhance
from configs.model_config import CHUNK_SIZE

from modelscope.pipelines import pipeline
# todo: 增加导入方法
from modelscope.utils.constant import Tasks
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(model_name=os.path.join(os.environ['HOME'], ".cache/huggingface/hub/models--GanymedeNil--text2vec-large-chinese"),
                                   model_kwargs={'device': EMBEDDING_DEVICE})


class TextCut:
    def __init__(self, min_len=20, step=10, stop_list=None):
        self.min_len = min_len  # 自定义最短长度
        self.step = step  # 自定义划窗步长
        if stop_list and isinstance(stop_list, list):
            self.stop_list = stop_list  # 自定义分割标点符
        else:
            self.stop_list = ['.', '!', '|', '。', '！', '；', ';', '?', '？', ',']
        self.split_patten = '[' + ''.join(self.stop_list) + ']'

    def find_now_index(self, now_point, sum_len_list):
        for i in range(len(sum_len_list) - 1):
            if now_point >= sum_len_list[i] and now_point < sum_len_list[i + 1]:
                return i + 1
        else:
            return 0

    def cut(self, text):
        if not isinstance(text, str):
            raise TypeError
        split_text = re.split(self.split_patten, text)
        len_list = np.array([len(x) for x in split_text])
        sum_len_list = np.cumsum(len_list)
        result_list = []
        end_point = 0
        pre_index = 0
        while end_point <= sum_len_list[-1]:
            end_point += self.step
            now_index = self.find_now_index(end_point, sum_len_list)
            if np.sum(len_list[pre_index:now_index]) >= self.min_len:
                result_list.append(''.join(split_text[pre_index:now_index]))
                pre_index = now_index

        if pre_index < len(split_text):
            last = ''.join(split_text[pre_index:])
            if len(last) >= self.min_len:
                result_list.append(last)
        return result_list


class lzc_ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = SENTENCE_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text1(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

    # def split_text(self, text: str) -> List[str]:   ##此处需要进一步优化逻辑
    #     if self.pdf:
    #         text = re.sub(r"\n{3,}", r"\n", text)
    #         text = re.sub('\s', " ", text)
    #         text = re.sub("\n\n", "", text)
    #
    #     text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
    #     text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
    #     text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
    #     text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
    #     # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    #     text = text.rstrip()  # 段尾如果有多余的\n就去掉它
    #     # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    #     ls = [i for i in text.split("\n") if i]
    #     for ele in ls:
    #         if len(ele) > self.sentence_size:
    #             ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
    #             ele1_ls = ele1.split("\n")
    #             for ele_ele1 in ele1_ls:
    #                 if len(ele_ele1) > self.sentence_size:
    #                     ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
    #                     ele2_ls = ele_ele2.split("\n")
    #                     for ele_ele2 in ele2_ls:
    #                         if len(ele_ele2) > self.sentence_size:
    #                             ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
    #                             ele2_id = ele2_ls.index(ele_ele2)
    #                             ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
    #                                                                                                    ele2_id + 1:]
    #                     ele_id = ele1_ls.index(ele_ele1)
    #                     ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]
    #
    #             id = ls.index(ele)
    #             ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
    #     return ls

    def split_text(self, text: str) -> List[str]:  # 此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        # print(f'原文：\n{text}')
        list_content = text.split('\n')
        temp_content = []
        for content in list_content:
            content = content.strip()
            # 判断是否存在段落结束符，若存在，不做处理，若不存在，则先在某尾处添加标识符“\n\n”， 后续会将该符号转换为“@lizhichao”，最后记得还原成“\n”
            if len(content) > 0:
                if content.endswith("。") or content.endswith("“") or content.endswith(".") or content.endswith('”'):
                    content = content
                else:
                    content = content + "\n\n"
                temp_content.append(content)
            else:
                continue

        str_content = "\n".join(temp_content)
        # print(f'使用换行符切分后的内容为：\n{str_content}')
        # 此处进行段落合并：1、没有段落结束符的小标题与后文段落进行合并；2、段落之间包含转折关系的两段进行合并。合并使用特殊标识符“@lizhichao”进行连接
        str_content = re.sub('\n\n\n', '@lizhichao', str_content)
        # print(f'使用特殊符号置换后的内容为：\n{str_content}')
        str_content = re.sub('\n\n', "", str_content)
        # print(f'当前内容为：\n{str_content}')
        str_content = re.sub("\n(?=而|但|对于|此外|因此|与此同时|这种|基于此|但是|然而)", "@lizhichao", str_content)
        new_content_list = str_content.split('\n')
        # 定义最终的返回对象
        final_content_list = []

        # todo: 最后对结果进行整合判断，先将特殊标记字符进行处理，将“@lizhichao”还原成“\n”，另外，对段落长度再次判断，当内容长度大于300时，使用滑动窗口方式进行切分得到子串
        for new_content in new_content_list:
            new_content = re.sub("@lizhichao", "\n", new_content)
            if len(new_content) >= 350:
                split_list = self.split_paragraph(text=new_content)
                final_content_list.extend(split_list)
            else:
                final_content_list.append(new_content)
        return final_content_list

    def split_paragraph(self, text: str) -> List[str]:
        text_cut_helper = TextCut(min_len=100, step=10, stop_list=['。'])
        # 将文本内容按照指定的分隔符分割成若干段落
        text_list = text_cut_helper.cut(text)
        return text_list


class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 如果使用模型进行文档语义切分，那么需要安装modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 考虑到使用了三个模型，可能对于低配置gpu不太友好，因此这里将模型load进cpu计算，有需要的话可以替换device为自己的显卡id
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        print("=====已进入切分方法=====")
        p = pipeline(
            task=Tasks.document_segmentation,
            model='damo/nlp_bert_document-segmentation_chinese-base',
            )

        # p = pipeline(
        #     config_file='/home/zzsn/.cache/modelscope/hub/damo/nlp_bert_document-segmentation_chinese-base/config.json',
        #     # task="document-segmentation",
        #     task=Tasks.document_segmentation,
        #     model='/home/zzsn/.cache/modelscope/hub/damo/nlp_bert_document-segmentation_chinese-base',
        #     device="cpu",
        #     preprocessor='document-segmentation')
        print("=====准备进行切分=====")
        result = p(documents=text)
        print(f'切分后的结果为：\n{result}')
        sent_list = []
        result_list = result["text"].split("\n\t")
        count = 0
        for i in range(len(result_list)):
            _a = result_list[i]
            _ = f'第{i+1}段，长度为{len(_a)}，内容为：{_a}'
            if len(_a) > 0:
                count += len(result_list[i])
                sent_list.append(_a)
        into_info = "\n".join(sent_list)
        # sent_list = [i for i in result["text"].split("\n\t") if i]
        # temp_text = []
        # for __ in sent_list:
        #     print(f'当前内容为：{__}')
        #     if len(__) > 0:
        #         modified_text = self.add_sep_to_paragraphs(text=text, target_paragraph=__)
        #         temp_text.append(modified_text)
        # taged_text = ''.join(temp_text)
        # print(taged_text)


        into_file = f'原文长度{len(text)}，内容为：\n【{text}】\n\n切分后总长度为：{count}，内容为：\n{into_info}\n\n'
        file.write(into_file)
        # print(f'sent_list 为： {sent_list}')

        return sent_list

    def add_sep_to_paragraphs(self, text: str, target_paragraph: str):
        # import re
        #
        # pattern = f"{target_paragraph.strip()}"  # 正则表达式模式
        # match = re.search(pattern, text)  # 在text中搜索匹配项
        #
        # if match:
        #     # 执行找到匹配项后的操作
        #     start_index = match.start()  # 获取匹配项的起始索引位置
        # else:
        #     # 处理未找到匹配项的情况
        #     print("未找到匹配项")
        #     return None
            # 执行其他适当的错误处理操作或提供
        start_index = text.index(target_paragraph.strip())  # 获取目标段落的开始索引
        # start_index, end_index, para = self.split_sentence(content=text, target_paragraph=target_paragraph)
        print(f'开始索引：{start_index}')
        end_index = start_index + len(target_paragraph)  # 获取目标段落的结束索引
        modified_text = text[:start_index] + "<sep>" + text[start_index:end_index] + "<sep>" + text[end_index:]
        print(f'添加分隔符后的内容为：{modified_text}')
        return modified_text


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def load_file(filepath, sentence_size=SENTENCE_SIZE, using_zh_title_enhance=ZH_TITLE_ENHANCE):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    # elif filepath.lower().endswith(".txt"):
    #     # 先利用TextLoader 将文件内容转换为List ，list里面是一个Document对象，该对象有两个属性，page_content、metadata
    #     loader = TextLoader(filepath, autodetect_encoding=True)
    #     logger.info(f'第一步实例化加载文本数据对象：{loader}')
    #     # 定义文本分割器
    #     textsplitter = lzc_ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
    #     logger.info(f'第二步实例化文本分割器对象：{textsplitter}')
    #     docs = loader.load_and_split(textsplitter)
    #     logger.info(f'第三步文本分割后的的结果为：{docs}')
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = lzc_ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = lzc_ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
        docs = loader.load()
    elif filepath.lower().endswith(".doc"):
        loader = Docx2txtLoader(filepath)
        # textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        textsplitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".docx"):
        loader = Docx2txtLoader(filepath)
        # textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        textsplitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".txt"):
        # 先利用TextLoader 将文件内容转换为List ，list里面是一个Document对象，该对象有两个属性，page_content、metadata
        loader = TextLoader(filepath, autodetect_encoding=True)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        # print(f'#'*50)
        # print(f'原文为：\n{text}')
        # print(f'#' * 50)
        logger.info(f'第一步实例化加载文本数据对象：{loader}')
        # 定义文本分割器
        textsplitter = AliTextSplitter(pdf=False)
        logger.info(f'第二步实例化文本分割器对象：{textsplitter}')
        docs = loader.load_and_split(textsplitter)
        logger.info(f'第三步文本分割后的的结果为：{docs}')
    else:
        # todo： 7月14号测试AliTextSplitter方法
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = lzc_ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    if using_zh_title_enhance:
        docs = zh_title_enhance(docs)
    write_check_file(filepath, docs)
    return docs


def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]


def init_knowledge_vector_store(
                                filepath: str or List[str],
                                vs_path: str or os.PathLike = None,
                                knowledge_base_id: str = None,
                                sentence_size=SENTENCE_SIZE):
    loaded_files = []
    failed_files = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None
        elif os.path.isfile(filepath):
            file = os.path.split(filepath)[-1]
            try:
                docs = load_file(filepath, sentence_size)
                logger.info(f"{file} 已成功加载")
                loaded_files.append(filepath)
            except Exception as e:
                logger.error(e)
                logger.info(f"{file} 未能成功加载")
                return None
        elif os.path.isdir(filepath):
            docs = []
            for fullfilepath, file in tqdm(zip(*tree(filepath, ignore_dir_names=['tmp_files'])), desc="加载文件"):
                try:
                    docs += load_file(fullfilepath, sentence_size)
                    loaded_files.append(fullfilepath)
                except Exception as e:
                    logger.error(e)
                    failed_files.append(file)

            if len(failed_files) > 0:
                logger.info("以下文件未能成功加载：")
                for file in failed_files:
                    logger.info(f"{file}\n")

    else:
        docs = []
        for file in filepath:
            try:
                docs += load_file(file)
                logger.info(f"{file} 已成功加载")
                loaded_files.append(file)
            except Exception as e:
                logger.error(e)
                logger.info(f"{file} 未能成功加载")

    print(f"docs 对象为：{docs}")

    if len(docs) > 0:
        print(f'过滤前长度{len(docs)}')
        new_docs = []
        for i in docs:
            if len(i.page_content) < 20:
                continue
            else:
                # print(f'当前内容为：{i.page_content}')
                new_docs.append(i)
        print(f'过滤后长度{len(new_docs)}')
        # todo: 0712 增加长度过滤
        print("文件加载完毕，正在生成向量库")

        vector_store = Milvus.from_documents(
            new_docs,
            embeddings,
            collection_name="{}".format(knowledge_base_id),
            connection_args={"host": "127.0.0.1", "port": "19530"},
        )

        torch_gc()
        print("=====向量生成完毕=====")

        # vector_store.save_local(vs_path)
        return vs_path, loaded_files, vector_store
    else:
        print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
        return None, loaded_files, None


if __name__ == "__main__":
    # file_path = "/home/zzsn/zhangtao/pycharm_projects/langchain/langchain-ChatGLM/knowledge_base/yjzx_news_vdb/content/--王志乐-坚定不移地在融入全球经济中打造世界一流企业_特稿.txt"

    # file_path = "/home/zzsn/zhangtao/pycharm_projects/langchain/langchain-ChatGLM/knowledge_base/yjzx_news_vdb/content/10000.txt"
    root_path = "/home/zzsn/zhangtao/pycharm_projects/langchain/langchain-ChatGLM/knowledge_base/yjzx_news_vdb/content/"
    with open('semantic_split_method_result.txt', 'a+', encoding='utf-8') as file:
        for dirpath, dirnames, filenames in os.walk(root_path):
            for f0 in tqdm(filenames[:2]):
                if f0.endswith('.txt'):
                    file_path = os.path.join(dirpath, f0)
                    vs_path = None
                    knowledge_base_id = "yjzx_news_vdb"
                    vs_path, loaded_files, _ = init_knowledge_vector_store([file_path], vs_path, knowledge_base_id)



    # p = pipeline(
    #     task=Tasks.document_segmentation,
    #     model='damo/nlp_bert_document-segmentation_chinese-base')
    #
    # result = p(
    #     documents='移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。'
    # )
    # print(result)
    # sent_list = [i for i in result["text"].split("\n\t") if i]
    # print(f'sent_list 为： {sent_list}')







