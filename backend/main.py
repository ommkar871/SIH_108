from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BIS-Sync API",
    description="Semantic search engine for Indian Standards",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
df = None
index = None
embeddings = None


@app.on_event("startup")
async def startup_event():
    
    global model, df, index, embeddings
    
    logger.info("BIS-Sync Starting Up...")
    
    # Load the embedding model
    logger.info(" Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    # Load BIS standards from CSV
    logger.info("Loading BIS standards from CSV...")
    csv_path = 'BIS_Standards.csv'
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV file not found at {csv_path}")
        raise FileNotFoundError(f"BIS_Standards.csv not found. Please ensure it's in the same directory as main.py")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded {len(df)} BIS standards")
    except Exception as e:
        logger.error(f"❌ Failed to load CSV: {e}")
        raise
    
    # Create embeddings for all standards
    logger.info("Creating embeddings for all standards (this may take 1-2 minutes)...")
    try:
        descriptions = df['DESCRIPTION'].tolist()
        embeddings = model.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)
        logger.info(f"✅ Created {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"❌ Failed to create embeddings: {e}")
        raise
    
    # Create FAISS index
    logger.info(" Building FAISS vector index...")
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        logger.info(f"✅ FAISS index ready ({dimension} dimensions, {len(embeddings)} vectors)")
    except Exception as e:
        logger.error(f"❌ Failed to create FAISS index: {e}")
        raise
    
    logger.info(" Backend initialized")
    logger.info("=" * 60)


@app.get("/")
def root():
    
    return {
        "message": "Welcome to BIS-Sync API",
        "endpoints": {
            "search": "/search?query=steel%20pipe%20water&top_k=3",
            "standards": "/standards",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "standards_loaded": len(df) if df is not None else 0,
        "model": "all-MiniLM-L6-v2",
        "backend": "FastAPI + FAISS"
    }

@app.get("/search")
def search(query: str, top_k: int = 3):
   
    # Verify input
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
    
    if model is None or index is None or df is None:
        raise HTTPException(status_code=503, detail="Backend not fully initialized")
    
    try:
        # Convert user query to embedding
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for rank, idx in enumerate(indices[0], 1):
            row = df.iloc[idx]
            distance = distances[0][rank - 1]
            
            match_score = float(max(0, 100 - (float(distance) * 25)))
            
            # Extract keywords from description
            description = row['DESCRIPTION']
            keywords = description.split()[:8]  
            keywords_str = ' '.join(keywords)
            
            results.append({
                "rank": rank,
                "is_code": row['IS_CODE'],
                "title": row['TITLE'],
                "description": row['DESCRIPTION'],
                "domain": row['DOMAIN'],
                "match_score": round(match_score, 1),
                "match_confidence": "High" if match_score >= 80 else "Medium" if match_score >= 60 else "Low",
                "explanation": f"Matched on keywords: {keywords_str}...",
                "keywords": row.get('KEYWORDS', 'N/A')
            })
        
        return {
            "query": query,
            "query_length": len(query),
            "results_count": len(results),
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/standards")
def get_all_standards(skip: int = 0, limit: int = 100):
    
    if df is None:
        raise HTTPException(status_code=503, detail="Standards not loaded")
    
    if limit > 100:
        limit = 100
    
    standards = df.iloc[skip:skip+limit].to_dict(orient='records')
    
    return {
        "total": len(df),
        "returned": len(standards),
        "skip": skip,
        "limit": limit,
        "standards": standards
    }

@app.get("/standards/{is_code}")
def get_standard_by_code(is_code: str):
    
    if df is None:
        raise HTTPException(status_code=503, detail="Standards not loaded")
    
    # Search for the code (case-insensitive)
    mask = df['IS_CODE'].str.upper() == is_code.upper()
    result = df[mask]
    
    if result.empty:
        raise HTTPException(status_code=404, detail=f"Standard {is_code} not found")
    
    return result.iloc[0].to_dict()

@app.get("/domains")
def get_domains():
    
    if df is None:
        raise HTTPException(status_code=503, detail="Standards not loaded")
    
    domain_counts = df['DOMAIN'].value_counts().to_dict()
    
    return {
        "total_domains": len(domain_counts),
        "domains": domain_counts,
        "domains_list": sorted(domain_counts.keys())
    }

@app.get("/search-by-domain")
def search_by_domain(query: str, domain: str, top_k: int = 3):
    
    if model is None or index is None or df is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    # Get all results first
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype('float32'), len(df))
    
    # Filter by domain
    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        if row['DOMAIN'].lower() == domain.lower():
            distance = distances[0][len(results)] if len(results) < len(indices[0]) else 999
            match_score = float(max(0, 100 - (float(distance) * 25)))
            results.append({
                "is_code": row['IS_CODE'],
                "title": row['TITLE'],
                "description": row['DESCRIPTION'],
                "domain": row['DOMAIN'],
                "match_score": round(match_score, 1)
            })
            if len(results) >= top_k:
                break
    
    return {
        "query": query,
        "domain_filter": domain,
        "results": results,
        "results_count": len(results)
    }

@app.get("/stats")
def get_stats():
    """
    📊 Get statistics about the BIS standards database
    
    Returns:
    --------
    Various statistics about loaded standards
    """
    
    if df is None:
        raise HTTPException(status_code=503, detail="Standards not loaded")
    
    return {
        "total_standards": len(df),
        "domains": df['DOMAIN'].nunique(),
        "domain_breakdown": df['DOMAIN'].value_counts().to_dict(),
        "avg_description_length": int(df['DESCRIPTION'].str.len().mean()),
        "is_codes": df['IS_CODE'].nunique(),
        "model_dimensions": 384,  
        "search_engine": "FAISS (Flat L2)"
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": True,
        "status_code": exc.status_code,
        "detail": exc.detail
    }



if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("BIS-Sync Backend Starting")
    print("=" * 60)
    print(" API URL: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print(" Make sure BIS_Standards.csv is in the same directory")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
