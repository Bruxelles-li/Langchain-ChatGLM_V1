import os
import getpass
#
# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from langchain.document_loaders import Docx2txtLoader
from configs.model_config import embedding_model_dict, EMBEDDING_MODEL, EMBEDDING_DEVICE


def demo(filepath):
    if filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=10)
        docs = text_splitter.split_documents(documents)
    elif filepath.lower().endswith(".doc"):
        loader = Docx2txtLoader(filepath)
        # textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        textsplitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=10)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".docx"):
        loader = Docx2txtLoader(filepath)
        # textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        textsplitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=10)
        docs = loader.load_and_split(textsplitter)

    else:
        loader = TextLoader(filepath)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=10)
        docs = text_splitter.split_documents(documents)


    # embeddings = OpenAIEmbeddings()

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],
                                       model_kwargs={'device': EMBEDDING_DEVICE})
    # docs = []
    vector_db = Milvus.from_documents(
        docs,
        embeddings,
        collection_name='test001',
        connection_args={"host": "127.0.0.1", "port": "19530"},
    )

    query = "数字化转型挑战"
    sim_docs = vector_db.similarity_search(query)
    sim_docs2 = vector_db.similarity_search_with_score(query)

    print(docs[0].page_content)


def similarity_search():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],
                                       model_kwargs={'device': EMBEDDING_DEVICE})
    vector_db = Milvus(
        embeddings,
        collection_name='yjzx_books3',
        connection_args={"host": "127.0.0.1", "port": "19530"},
    )
    while True:
        print('User: ', end='')
        query = input()
        # query = "数字化转型挑战"
        # sim_docs = vector_db.similarity_search(query)
        sim_docs2 = vector_db.similarity_search_with_score(query)
        print('Robot: {}'.format(''))
        for i in sim_docs2:
            print('score: {}, page_content: {}\n\n'.format(i[1], i[0].page_content))


def create_col(collection_name):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],
                                       model_kwargs={'device': EMBEDDING_DEVICE})
    vector_db = Milvus(
        embeddings,
        collection_name=collection_name,
        connection_args={"host": "127.0.0.1", "port": "19530"},
    )


def clean_col():
    from pymilvus import connections

    conn = connections.connect("yjzx_books", host="127.0.0.1", port=19530)
    conn.clean()
    # database = db.create_database

# clean_col()
# create_col()
# demo('state_of_the_union.txt')
# demo('wKjIbGSmpsSAAqe9AB4lwEUh0v8008.doc')
# create_col('yjzx_books3')
similarity_search()
