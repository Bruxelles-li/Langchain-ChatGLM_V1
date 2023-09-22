import requests
import json
import pymysql
import datetime
from tqdm import tqdm


def query_book_info():
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
    mysql_conn = pymysql.connect(host='114.115.185.13',
                                 port=3305,
                                 user='root',
                                 password='sc24&bgqsc',
                                 database='clb_project')
    books_info = []
    try:
        with mysql_conn.cursor() as cursor:
            sql_str = f'''SELECT id, name, book_isbn, site_isbn, author, 
            publishing_house, publish_date, doc_page_size, clc_type_id, clc_type_name, 
            pdf_path, word_path, create_by, create_time FROM ai_report_book 
            where word_path is not null'''
            print('sql_str={}'.format(sql_str))
            cursor.execute(sql_str)
            rows = cursor.fetchall()
            for row in list(rows):
                books_info.append({
                    "id": row[0],
                    "name": row[1],
                    "book_isbn": row[2],
                    "site_isbn": row[3],
                    "author": row[4],
                    "publishing_house": row[5],
                    "publish_date": row[6],
                    "doc_page_size": row[7],
                    "clc_type_id": row[8],
                    "clc_type_name": row[9],
                    "pdf_path": row[10],
                    "word_path": row[11],
                    "create_by": row[12],
                    "create_time": row[13].strftime('%Y-%m%d %H:%M:%S')
                })

        cursor.close()

        return books_info
    except Exception as e:
        print('Error: {}'.format(e))
        return books_info


print('检索book ...')
books_info = query_book_info()
print('检索完成！')
print('开始入向量数据库 ...')

url = "http://localhost:7861/book/upload_book"

for book in tqdm(books_info):
    payload = json.dumps({
        "knowledge_base_id": "yjzx_books_vdb",
        "id": book['id'],
        "name": book['name'],
        "no_isbn": book['book_isbn'],
        "site_isbn": book['site_isbn'],
        "author": book['author'],
        "publishing_house": book['publishing_house'],
        "publish_date": book['publish_date'],
        "page_size": book['doc_page_size'],
        "clc_type_id": book['clc_type_id'],
        "clc_type_name": book['clc_type_name'],
        "filePath": book['word_path'],
        "create_by": book['create_by'],
        "create_time": str(book['create_time'])
    })

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


