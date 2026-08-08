import asyncio
import time
from fastapi import FastAPI, HTTPException, Request
import json
import uvicorn
from typing import Dict
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import aiohttp
from fastapi.responses import JSONResponse
from pathlib import Path
import re
import sys
from models import (
    SearchRequest,
    ReadPdfInfo,
    FetchWebContent,
    WebParseInfo,
    BatchSearchAndFilterInfo,
    GenerateKeywordsInfo,
    CheckConditionInfo,
)
from api_utils.web_search_api import serper_google_search
from api_utils.pdf_read_api import read_pdf_from_url
from api_utils.fetch_web_page_api import fetch_web_content
import requests
import os


app = FastAPI()

# Initialize in-memory rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

CURRENT_DIR = Path(__file__).resolve().parent
BASE_TOOL_DIR = CURRENT_DIR.parent / "MCP" / "server" / "BASE-TOOL-Server"

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default

DEFAULT_TIMEOUT = _env_float("BROWSE_MASTER_TIMEOUT", 300.0)
WEB_PARSE_TIMEOUT = _env_float("BROWSE_MASTER_WEB_PARSE_TIMEOUT", DEFAULT_TIMEOUT)
LLM_ENDPOINT_TIMEOUT = _env_float("BROWSE_MASTER_LLM_ENDPOINT_TIMEOUT", DEFAULT_TIMEOUT)


def _load_web_agent_config() -> dict:
    config_path = CURRENT_DIR.parent / "configs" / "web_agent.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


async def _llm_call(prompt: str, model: str | None = None) -> str:
    if str(BASE_TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_TOOL_DIR))
    from utils.llm_caller import llm_call

    web_config = _load_web_agent_config()
    model_to_use = model or web_config.get("USE_MODEL") or web_config.get("BASE_MODEL")
    if not model_to_use:
        raise ValueError("No LLM model configured for web_parse")
    return await asyncio.wait_for(llm_call(prompt, model_to_use), timeout=LLM_ENDPOINT_TIMEOUT)


def _fallback_keywords(seed_keyword: str) -> list[str]:
    cleaned = " ".join(seed_keyword.split())
    keywords = [cleaned]
    if len(cleaned) <= 180:
        keywords.append(f'"{cleaned}"')
    keywords.extend([
        f"{cleaned} source",
        f"{cleaned} evidence",
    ])
    seen = set()
    return [k for k in keywords if k and not (k in seen or seen.add(k))][:8]


async def _generate_keywords(seed_keyword: str) -> list[str]:
    prompt = (
        "Generate 5 concise web search queries for the following target. "
        "Return only JSON: {\"keywords\": [\"...\"]}.\n\n"
        f"Target: {seed_keyword}"
    )
    try:
        response = await _llm_call(prompt)
        data = _extract_json_object(response) or {}
        keywords = data.get("keywords", [])
        if isinstance(keywords, list):
            keywords = [str(k).strip() for k in keywords if str(k).strip()]
            if keywords:
                return keywords[:8]
    except Exception:
        pass
    return _fallback_keywords(seed_keyword)


async def _check_condition(content: str, condition: str) -> str:
    prompt = (
        "Decide whether the content satisfies the condition. "
        "Answer with exactly one word: yes, no, or unknown.\n\n"
        f"Condition: {condition}\n\nContent:\n{content[:6000]}"
    )
    try:
        response = (await _llm_call(prompt)).strip().lower()
        match = re.search(r"\b(yes|no|unknown)\b", response)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"

@app.post("/search")
@limiter.limit("200/second")
async def search(request: Request, search_request: SearchRequest):
    try:
        result = await serper_google_search(
            search_request.query, 
            search_request.serper_api_key, 
            search_request.top_k, 
            search_request.region, 
            search_request.lang, 
            depth=search_request.depth
        )
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    


@app.post("/read_pdf")
@limiter.limit("200/second")
async def read_pdf(request: Request, read_pdf_request: ReadPdfInfo):
    try:
        result = await read_pdf_from_url(read_pdf_request.url)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/fetch_web")
@limiter.limit("200/second")
async def fetch_web(request: Request, fetch_web_request: FetchWebContent):
    try:
        result = await fetch_web_content(fetch_web_request.url)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/web_parse")
@limiter.limit("200/second")
async def web_parse(request: Request, web_parse_request: WebParseInfo):
    try:
        target_url = web_parse_request.target_url
        if not target_url:
            raise HTTPException(status_code=400, detail="Missing link/url")

        if str(BASE_TOOL_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_TOOL_DIR))
        from web_agent.web_parse import parse_htmlpage

        result = await asyncio.wait_for(
            parse_htmlpage(
                target_url,
                web_parse_request.user_prompt,
                llm=web_parse_request.llm,
            ),
            timeout=WEB_PARSE_TIMEOUT,
        )
        return result
    except HTTPException as e:
        raise e
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"web_parse timed out after {WEB_PARSE_TIMEOUT:g}s")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/generate_keywords")
@limiter.limit("200/second")
async def generate_keywords(request: Request, generate_request: GenerateKeywordsInfo):
    try:
        keywords = await _generate_keywords(generate_request.seed_keyword)
        return {"keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/check_condition")
@limiter.limit("200/second")
async def check_condition(request: Request, check_request: CheckConditionInfo):
    try:
        is_relevant = await _check_condition(check_request.content, check_request.condition)
        return {"is_relevant": is_relevant}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/batch_search_and_filter")
@limiter.limit("200/second")
async def batch_search_and_filter(request: Request, batch_request: BatchSearchAndFilterInfo):
    try:
        web_config = _load_web_agent_config()
        keywords = await _generate_keywords(batch_request.keyword)
        yes = []
        unknown = []
        seen_links = set()

        for keyword in keywords[:5]:
            result = await serper_google_search(
                keyword,
                web_config.get("serper_api_key", ""),
                batch_request.top_k,
                web_config.get("search_region", "us"),
                web_config.get("search_lang", "en"),
            )
            for item in result.get("organic", []) if isinstance(result, dict) else []:
                link = item.get("link")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                content = f"{item.get('title', '')}\n{item.get('snippet', '')}"
                verdict = await _check_condition(content, batch_request.keyword)
                if verdict == "yes":
                    yes.append(item)
                elif verdict == "unknown":
                    unknown.append(item)

        return {"keywords": keywords, "yes": yes, "unknown": unknown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Limit is 200 requests per second."},
        headers={"Retry-After": "1"}
    )




if __name__ == "__main__":
    import os

    PORT = os.getenv('PORT', 1234)

    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0", 
        port=int(PORT),
        lifespan="on",
        workers=1
    )
