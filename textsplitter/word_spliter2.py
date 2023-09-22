from docx import Document

# 定义标题样式列表
title_styles = ['Heading 1', 'Heading 2', 'Heading 3', 'Heading 4', 'Heading 5', 'Heading 6']

# 打开Word文档
doc = Document('/home/zzsn/zhangtao/pycharm_projects/langchain/langchain-ChatGLM/knowledge_base/国资国企研究知识库-test/content/wKjIbGSmpr2ABmfyABXCmmR9Zyo746.doc')

# 遍历文档中的段落和章节
chapters = []
current_chapter = None

for paragraph in doc.paragraphs:
    # 获取段落样式
    style = paragraph.style.name
    print(style)
    # 判断段落样式是否为标题样式
    if style in title_styles:
        if current_chapter is not None:
            chapters.append(current_chapter)

        # 获取标题级别
        level = title_styles.index(style) + 1

        current_chapter = {
            'title': paragraph.text,
            'level': level,
            'paragraphs': []
        }
    else:
        if current_chapter is not None:
            current_chapter['paragraphs'].append(paragraph.text)

# 添加最后一个章节
if current_chapter is not None:
    chapters.append(current_chapter)

# 输出结果
for chapter in chapters:
    print('Chapter:', chapter['title'])
    for paragraph in chapter['paragraphs']:
        print('Paragraph:', paragraph)
    print()
