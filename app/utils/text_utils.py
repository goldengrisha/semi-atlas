import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def collect_links(url, base_url, max_depth, depth: int = 0, visited: set = set()):
    if depth > max_depth:
        return []

    links = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link.get("href")
            full_url = urljoin(base_url, href)
            if (
                full_url not in visited
                and urlparse(full_url).netloc == urlparse(base_url).netloc
            ):
                visited.add(full_url)
                links.append(full_url)
                links.extend(
                    collect_links(full_url, base_url, max_depth, depth + 1, visited)
                )

    except requests.RequestException as e:
        print(f"Request failed: {e}")

    return links


def load_text_from_url(urls: list[str]) -> list[Document]:
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def get_in_memory_documents(
    documents: list[Document], embeddings_model: HuggingFaceEmbeddings
) -> DocArrayInMemorySearch:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splitted_documents = text_splitter.split_documents(documents)

    db = DocArrayInMemorySearch.from_documents(splitted_documents, embeddings_model)

    return db
