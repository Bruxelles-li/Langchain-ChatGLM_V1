from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch
from langchain.document_loaders import TextLoader
from configs.model_config import embedding_model_dict, EMBEDDING_MODEL, EMBEDDING_DEVICE


loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# embedding = OpenAIEmbeddings()
embedding = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],
                                  model_kwargs={'device': EMBEDDING_DEVICE})

elastic_host = "114.115.215.250"
username, password = 'elastic', 'zzsn9988'
elasticsearch_url = f"http://{username}:{password}@{elastic_host}:9243"
elastic_vector_search = ElasticVectorSearch(
    elasticsearch_url=elasticsearch_url,
    index_name="test_index20230703",
    embedding=embedding
)
elastic_vector_search.add_documents(docs)
db = elastic_vector_search

# db = ElasticVectorSearch.from_documents(
#     docs,
#     embedding=embedding,
#     elasticsearch_url=elasticsearch_url,
#     index_name="test_index20230703",
#     async_timeout=20
# )
#
query = "如果 AI 只是被提示100100/40056等于几？"
docs = db.similarity_search(query)

print(docs[0].page_content)