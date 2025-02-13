import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader #文档加载器的一种，用于加载网页内容
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # 向量数据库，用于存储向量数据
from langchain.text_splitter import RecursiveCharacterTextSplitter # 用于文档分割
 # set the LANGSMITH_API_KEY environment variable (create key in settings)
from langchain import hub
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain # 用于创建LCEL Runnable object, 将llm和prompt链接在一起
from langchain.chains import create_retrieval_chain

load_dotenv()

# 搜索网站上的信息
url="https://www.langchain.com/"
loader = WebBaseLoader(url)

document = loader.load()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

prompt = hub.pull("langchain-ai/retrieval-qa-chat")
llm = ChatGroq(
    temperature=0.8,
    model="llama3-70b-8192",
    groq_api_key = os.getenv("GROQ_API_KEY")
)

query = "What is langchian?"

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
combine_docs_chain.invoke({
    "input": query,
    "context": docs
})

retriever = FAISS.load_local(
    "vector_db", 
    embeddings_model,
    allow_dangerous_deserialization=True
).as_retriever()

retriever_chian = create_retrieval_chain(retriever, combine_docs_chain)
response = retriever_chian.invoke({
    "input": query
})

print(response)
