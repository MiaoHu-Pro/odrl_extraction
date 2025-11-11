from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# load_dotenv()
import os

# 1
#
loader = PyPDFLoader("./data_sharing_agreement_1/DSA_0001.pdf")
pages = loader.load_and_split()

print(pages[4].page_content)

# print(f"第0页：\n{pages[2]}") ## 也可通过 pages[0].page_content只获取本页内容


# 2

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
#
# loader = PyPDFLoader("./data_sharing_agreement_1/DSA_0001.pdf")
# docs = loader.load()  # one Document per page; .page_content + .metadata
#
# # Split for embedding/RAG
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1200, chunk_overlap=200,
#     add_start_index=True,  # stores start idx in metadata
# )
# chunks = splitter.split_documents(docs)
#
# # Inspect
# print(len(docs), "pages ->", len(chunks), "chunks")
# print(chunks[0].metadata)  # {'source': '...pdf', 'page': 0, 'start_index': ...}
