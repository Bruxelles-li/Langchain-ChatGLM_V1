import requests
from urllib.parse import urlsplit


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
        with open('file.pdf', 'wb') as f:
            f.write(response.content)
        print('文件下载成功')

        return response.content, filename
    else:
        print('下载失败，状态码为', response.status_code)

        return None, None


_, _filename = file_download(url='https://www.guizhou.gov.cn/zwgk/rsxx/sydwgkzp/202306/P020230627609639286577.docx')
print(_filename)