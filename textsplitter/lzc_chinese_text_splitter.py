from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
import numpy as np
from configs.model_config import SENTENCE_SIZE


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
                result_list.append('。'.join(split_text[pre_index:now_index]) + '。')
                pre_index = now_index

        if pre_index < len(split_text):
            last = '。'.join(split_text[pre_index:])
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
            # 判断是否存在段落结束符，若存在，不做处理，若不存在，则先在某尾处添加标识符“\n\n”， 后续会将该符号转换为“<sep>”，最后记得还原成“\n”
            if len(content) > 0:
                if content.endswith("。") or content.endswith(".") or content.endswith('”'):
                    content = content
                else:
                    content = content + "\n\n"
                temp_content.append(content)
            else:
                continue

        str_content = "\n".join(temp_content).strip()
        # print(f'使用换行符切分后的内容为：\n{str_content}')
        # 此处进行段落合并：1、没有段落结束符的小标题与后文段落进行合并；2、段落之间包含转折关系的两段进行合并。合并使用特殊标识符“<sep>”进行连接
        str_content = re.sub('\n\n\n', '<sep>', str_content)
        # print(f'当前内容为：\n{str_content}')
        str_content = re.sub("\n(?=而|但|对于|此外|因此|与此同时|这种|基于此|但是|然而)", "<sep>", str_content)
        new_content_list = str_content.split('\n')
        # 定义最终的返回对象
        final_content_list = []

        # todo: 最后对结果进行整合判断，先将特殊标记字符进行处理，将“<sep>”还原成“\n”，另外，对段落长度再次判断，当内容长度大于300时，使用滑动窗口方式进行切分得到子串
        for new_content in new_content_list:
            new_content = re.sub("<sep>", "\n", new_content)
            if len(new_content) >= self.sentence_size:
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


