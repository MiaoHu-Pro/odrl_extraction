from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
import os

loader = PyPDFLoader("llama2.pdf")
pages = loader.load_and_split()

print(len(pages))

print(f"第0页：\n{pages[0]}") ## 也可通过 pages[0].page_content只获取本页内容
