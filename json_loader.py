import json
from langchain_core.documents import Document

def load_cv_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}: ")
        elif isinstance(obj, list):
            for item in obj:
                walk(item, prefix)
        else:
            docs.append(
                Document(page_content=f"{prefix}{obj}")
            )

    walk(data)
    return docs
