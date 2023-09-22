#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : smi_app.py
# @Time    : 2022/6/22 16:50
# @Author  : bruxelles_li
# @Software: PyCharm
import json
import os
import sys
sys.path.append("../../")
import logging
import argparse
from flask import Flask, request
from pathlib import Path
from image_loader import main

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %('
                                               'message)s')
logger = logging.getLogger(__name__)

# todo: 定义缓存路径
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "samples", "content", "中央公告")
cache_path = "../knowledge_base/samples/content/中央公告"
Path(cache_path).mkdir(parents=True, exist_ok=True)



app = Flask(__name__)

# 处理中文编码
app.config['JSON_AS_ASCII'] = False

# 跨域支持1
from flask_cors import CORS

CORS(app, supports_credentials=True)


@app.route("/", methods=["GET"])
def hello_world():
    app.logger.info('Hello World!')
    return "Hello World"


@app.route('/ocr_process/', methods=['GET', 'POST'])
def ocr_process():
    try:
        # 定义接收参数
        file = request.files['multiRequest']
        if file:
            file.save(os.path.join(cache_path, file.filename))
            res = main(image_file=file.filename)
        else:
            res = "文件内容有误"
        dict_result = {'code': 200,
                      'log': None,
                      'result': res
                      }

    except Exception as e:
        dict_result = {'code': 500,
                      'log': str(e),
                      'result': None
                      }
    logger.info(dict_result)
    return json.dumps(dict_result, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-port', dest='port', help='', default=8012)
    parser.add_argument('-host', dest='host', help='', default='0.0.0.0')
    args = parser.parse_args()
    app.run(host=args.host, port=int(args.port))
