"""Loader that loads image files."""
import sys
sys.path.append("../../")
from typing import List
from pdf_loader import UnstructuredPaddlePDFLoader

from langchain.document_loaders.unstructured import UnstructuredFileLoader
from paddleocr import PaddleOCR
import os
import nltk
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


class UnstructuredPaddleImageLoader(UnstructuredFileLoader):
    """Loader that uses unstructured to load image files, such as PNGs and JPGs."""

    def _get_elements(self) -> List:
        def image_ocr_txt(filepath, dir_path="../tmp_files"):
            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)
            if not os.path.exists(full_dir_path):
                os.makedirs(full_dir_path)
            filename = os.path.split(filepath)[-1]
            ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, show_log=False)
            result = ocr.ocr(img=filepath)

            ocr_result = [i[1][0] for line in result for i in line]
            txt_file_path = os.path.join(full_dir_path, "%s.txt" % (filename))
            with open(txt_file_path, 'w', encoding='utf-8') as fout:
                fout.write("\n".join(ocr_result))
            return txt_file_path

        txt_file_path = image_ocr_txt(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(filename=txt_file_path, **self.unstructured_kwargs)


def main(image_file: str):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "samples", "content", "中央公告")
    # image_file = "sample.pdf"
    if not image_file.endswith(".pdf"):
        print(image_file)
        loader = UnstructuredPaddleImageLoader(path + '/' + image_file, mode="elements")
    else:
        print(image_file)
        loader = UnstructuredPaddlePDFLoader(path + '/' + image_file, mode="elements")
    docs = loader.load()
    res = ''
    for doc in docs:
        res += ' '
        res += doc.page_content
    res = res.strip()
    print(res)
    return res
      
      
if __name__ == "__main__":
    print(main())
    # import sys
    #
    # sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    # path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "samples", "content", "中央公告")
    #
    # all_image = []
    # for file_name in os.listdir(path):
    #     all_image.append(file_name)
    #
    # for image_file in all_image:
    #     if not image_file.endswith(".pdf"):
    #         print(image_file)
    #         loader = UnstructuredPaddleImageLoader(path + '/' + image_file, mode="elements")
    #     else:
    #         print(image_file)
    #         loader = UnstructuredPaddlePDFLoader(path + '/' + image_file, mode="elements")
    #     docs = loader.load()
    #     res = ''
    #     for doc in docs:
    #         res += ' '
    #         res += doc.page_content
    #     res = res.strip()
    #     print(res)


        # print(doc)
