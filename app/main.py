from fastapi import FastAPI, HTTPException
from utils.db_utils import get_db
from models import SearchQuery


# Get DBF
db = get_db()

# Initialize FastAPI app
app = FastAPI()


@app.post("/search")
async def search_articles(search_query: SearchQuery):
    articles = db.similarity_search(search_query.query, k=10)

    if len(articles) == 0:
        raise HTTPException(status_code=404, detail="No articles found")

    return articles


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
