# app.py
# -*- coding: utf-8 -*-
import os
import re
import time
import functools
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


# ====== 載入 .env（若有）======
load_dotenv(override=True)

# ====== FastAPI 實例 ======
app = FastAPI(title="Books RAG ReadOnly API", version="1.0.0")

# ====== 全域設定（可用環境變數覆寫）======
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "at-rag-01")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "tz_books")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
SRC_DIM    = int(os.getenv("SRC_DIM", 384))   # 供參考
DST_DIM    = int(os.getenv("DST_DIM", 1536))  # Pinecone index 維度

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 或改成 ["http://127.0.0.1:5500", "http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 全域客戶端（啟動後才初始化）======
EMBEDDER = None
INDEX = None

# ====== 型別定義 ======
class SearchRequest(BaseModel):
    query: str = Field(..., description="使用者查詢文字")
    top_k_chunks: int = Field(50, ge=1, le=200, description="從 Pinecone 取回的 chunk 數量")
    final_n: int = Field(10, ge=1, le=50, description="最終返回的書籍數量")
    category_name: Optional[str] = Field(None, description="如: 佛教")
    publisher_name: Optional[str] = Field(None, description="如: 遠見天下文化出版股份有限公司")
    vector_types: Optional[List[str]] = Field(None, description="如: ['preface','catalogue']")
    keyword_weights: Optional[List[float]] = Field(None, description="[title_w, cat_pub_w, body_w]")

class SearchItem(BaseModel):
    score: float
    base_score: Optional[float] = None
    title: Optional[str] = None
    author: Optional[str] = None
    cat: Optional[str] = None
    publisher: Optional[str] = None
    isbn: Optional[str] = None
    prod_id: Optional[str] = None
    org_prod_id: Optional[str] = None
    vector_type: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    count: int
    items: List[SearchItem]

# ====== 工具函式 ======
def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def pad_to(v: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.shape[-1] == dim:
        return v
    if v.shape[-1] > dim:
        raise ValueError("查詢向量維度大於索引維度")
    pad = np.zeros((dim - v.shape[-1],), dtype=np.float32)
    return np.concatenate([v, pad], axis=0)

def embed_query_local(text: str) -> List[float]:
    """用本地開源模型嵌入查詢並 pad 到 DST_DIM。"""
    if EMBEDDER is None:
        raise HTTPException(status_code=503, detail="Embedder not ready")
    v = EMBEDDER.encode([text], normalize_embeddings=True)[0]
    v = pad_to(v, DST_DIM)
    return v.tolist()

def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\u3000]+", "", s)
    s = re.sub(r"[，,。．.!！?？:：;；\-–—_~·•．·･·]+", "", s)
    return s

def pick_book_key(md: dict) -> str:
    """決定書籍聚合鍵：org_prod_id > main_isbn > prod_id > (title,publisher) > chunk_id"""
    for k in ("org_prod_id", "main_isbn", "prod_id"):
        v = md.get(k)
        if v:
            return f"{k}::{v}"
    t = norm_text(md.get("prod_title_main") or "")
    p = norm_text(md.get("publisher_name") or "")
    if t or p:
        return f"titlepub::{t}::{p}"
    return f"noid::{md.get('chunk_id')}"

def keyword_bonus(query: str, md: dict, w=(0.08, 0.03, 0.01)) -> float:
    """簡易關鍵字加權：標題>分類/出版社>內文"""
    q = (query or "").lower()
    title = (md.get("prod_title_main") or "").lower()
    cat   = (md.get("cat4xsx_cat_nm") or "").lower()
    pub   = (md.get("publisher_name") or "").lower()
    body  = (md.get("chunk_text") or md.get("text") or "").lower()

    def hits(q, s):  # 粗略：以子字串計數
        return sum(1 for t in q.split() if t and t in s)

    return w[0]*hits(q,title) + w[1]*(hits(q,cat)+hits(q,pub)) + w[2]*hits(q,body)

def aggregate_by_book(res: Dict[str, Any], query: str, max_books: int = 50,
                      weights=(0.08,0.03,0.01)) -> List[Dict[str, Any]]:
    """把 chunk 級結果聚合為書籍級（取該書最高分＋關鍵字加權）"""
    buckets: Dict[str, Dict[str, Any]] = {}
    for m in res.get("matches", []):
        md   = m.get("metadata", {}) or {}
        key  = pick_book_key(md)
        base = m["score"]
        bonus = keyword_bonus(query, md, w=weights)
        score = base + bonus
        item = {
            "score": score,
            "base_score": base,
            "title": md.get("prod_title_main"),
            "author": md.get("main_author") or "",
            "cat": md.get("cat4xsx_cat_nm"),
            "publisher": md.get("publisher_name"),
            "isbn": md.get("main_isbn"),
            "prod_id": md.get("prod_id"),
            "org_prod_id": md.get("org_prod_id"),
            "vector_type": md.get("vector_type"),
        }
        if key not in buckets or score > buckets[key]["score"]:
            buckets[key] = item
    return sorted(buckets.values(), key=lambda x: x["score"], reverse=True)[:max_books]

def diversify_by_title_or_cat(books: List[Dict[str, Any]], target_n: int = 10) -> List[Dict[str, Any]]:
    """在書籍層級做多樣化：同 title 或同 cat 只留一筆，並遞補。"""
    if not books:
        return []
    selected: List[Dict[str, Any]] = []
    seen_titles, seen_cats = set(), set()
    pool = sorted(books, key=lambda x: x["score"], reverse=True)

    # 1) 先做 title 去重（最重要）
    for b in pool:
        t = norm_text(b.get("title"))
        if t and t not in seen_titles:
            selected.append(b)
            seen_titles.add(t)
        if len(selected) >= target_n:
            return selected

    # 2) 針對仍未選入的，再做 cat 去重
    for b in pool:
        if b in selected: 
            continue
        t = norm_text(b.get("title"))
        c = norm_text(b.get("cat"))
        if t and t in seen_titles:
            continue
        if c and c not in seen_cats:
            selected.append(b)
            seen_titles.add(t)
            seen_cats.add(c)
        if len(selected) >= target_n:
            return selected

    # 3) 仍不足則遞補（允許 cat 重複，但仍不重複 title）
    for b in pool:
        if b in selected:
            continue
        t = norm_text(b.get("title"))
        if t and t in seen_titles:
            continue
        selected.append(b)
        seen_titles.add(t)
        if len(selected) >= target_n:
            return selected
    return selected

# ====== 輕量快取（Pinecone Query 30 秒）======
def memoize_ttl(ttl_sec=30):
    def deco(fn):
        cache: Dict[str, Any] = {}
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, str(args), str(sorted(kwargs.items())))
            now = time.time()
            if key in cache:
                ts, val = cache[key]
                if now - ts < ttl_sec:
                    return val
            val = fn(*args, **kwargs)
            cache[key] = (now, val)
            return val
        return wrapper
    return deco

@memoize_ttl(ttl_sec=30)
def pinecone_query(vector: List[float], top_k: int, flt: Optional[Dict[str, Any]]):
    if INDEX is None:
        raise HTTPException(status_code=503, detail="Index not ready")
    return INDEX.query(namespace=NAMESPACE, vector=vector, top_k=top_k, include_metadata=True, filter=flt)

# ====== 啟動後才載入外部資源（Lazy Init）======
@app.on_event("startup")
def init_clients():
    import logging
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logging.error("PINECONE_API_KEY 未設定，僅提供 /health 錯誤回覆")
        return
    try:
        from sentence_transformers import SentenceTransformer
        from pinecone import Pinecone
        global EMBEDDER, INDEX
        EMBEDDER = SentenceTransformer(MODEL_NAME)
        INDEX = Pinecone(api_key=api_key).Index(INDEX_NAME)
        logging.info("Pinecone & Embedder 初始化完成")
    except Exception as e:
        logging.exception("初始化失敗：%s", e)

# ====== Endpoints ======
@app.get("/health")
def health():
    if INDEX is None:
        return {"ok": False, "reason": "INIT_NOT_READY_OR_NO_API_KEY"}
    try:
        stats = INDEX.describe_index_stats()
        ns = stats.get("namespaces", {}).get(NAMESPACE, {})
        return {"ok": True, "namespace": NAMESPACE, "vector_count": ns.get("vector_count", 0)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if EMBEDDER is None or INDEX is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    # 1) 向量化
    qvec = embed_query_local(req.query)

    # 2) 組合 filter
    clauses: List[Dict[str, Any]] = []
    if req.category_name:
        clauses.append({"cat4xsx_cat_nm": {"$eq": req.category_name}})
    if req.publisher_name:
        clauses.append({"publisher_name": {"$eq": req.publisher_name}})
    if req.vector_types:
        clauses.append({"vector_type": {"$in": req.vector_types}})
    flt = None if not clauses else (clauses[0] if len(clauses) == 1 else {"$and": clauses})

    # 3) 查 Pinecone（chunk 級）
    res = pinecone_query(qvec, top_k=req.top_k_chunks, flt=flt)

    # 4) 聚合（書籍級）
    weights = tuple(req.keyword_weights) if req.keyword_weights else (0.08, 0.03, 0.01)
    books_pool = aggregate_by_book(res, req.query, max_books=req.top_k_chunks, weights=weights)

    # 5) 去重多樣化
    diversified = diversify_by_title_or_cat(books_pool, target_n=req.final_n)

    return SearchResponse(
        query=req.query,
        count=len(diversified),
        items=[SearchItem(**b) for b in diversified]
    )

# 兼容命名：/recommend 等同 /search
@app.post("/recommend", response_model=SearchResponse)
def recommend(req: SearchRequest):
    return search(req)
