
prompt_templete4summary = '''
已知信息：
{context} 

根据上述已知信息，生成文章摘要，摘要内容尽可能是原文中的原话，不允许添加编造。 摘要字数限制是：{summary_max_length}字。
'''

prompt_templete4main_company = '''
已知信息：
{context} 

根据上述已知信息，回答这篇文章主要是哪些企业的事，不允许添加编造。 只回答企业名称。
'''
