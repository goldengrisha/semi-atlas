from langchain_community.vectorstores import DocArrayInMemorySearch

from utils.text_utils import (
    get_embedding_model,
    get_in_memory_documents,
    load_text_from_url,
    collect_links,
)


def get_db() -> DocArrayInMemorySearch:
    root_url = "https://help.atlas.so/"
    all_links = collect_links(root_url, root_url, 6)
    only_article_links = [link for link in all_links if "articles" in link]
    documents = load_text_from_url(only_article_links)
    embeddings_model = get_embedding_model()

    db = get_in_memory_documents(documents=documents, embeddings_model=embeddings_model)

    return db
