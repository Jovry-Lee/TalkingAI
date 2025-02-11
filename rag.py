import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader #文档加载器的一种，用于加载网页内容
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # 向量数据库，用于存储向量数据
from langchain.text_splitter import RecursiveCharacterTextSplitter # 用于文档分割
 
load_dotenv()

# 搜索网站上的信息
url="https://www.langchain.com/"
loader = WebBaseLoader(url)

document = loader.load()
# print(document)

# 文档分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # 分块大小
    chunk_overlap=200, # 允许分块间有重叠
    length_function=len, # 使用len方法进行分割
    separators=["\n\n", "\n", " ", ""] # 分割标志符
)

docs = text_splitter.split_documents(document)
# print(len(docs))

# 将分块数据从文字转换为embedding格式，方便llm可以辨认的格式
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2" # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
)

vectorDB = FAISS.from_documents(docs, embeddings_model)
vectorDB.save_local("vector_db")

