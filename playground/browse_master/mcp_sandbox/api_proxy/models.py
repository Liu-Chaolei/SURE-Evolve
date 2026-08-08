from typing import Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    serper_api_key: str = ""
    top_k: int = 10
    region: str = "us"
    lang: str = "en"
    depth: int = 0



class SearchPaperInfo(BaseModel):
    query:str


class ReadPdfInfo(BaseModel):
    url:str


class FetchWebContent(BaseModel):
    url:str   


class WebParseInfo(BaseModel):
    link: Optional[str] = None
    url: Optional[str] = None
    user_prompt: str = ""
    llm: str = ""

    @property
    def target_url(self) -> str:
        return self.link or self.url or ""


class BatchSearchAndFilterInfo(BaseModel):
    keyword: str
    top_k: int = 5


class GenerateKeywordsInfo(BaseModel):
    seed_keyword: str


class CheckConditionInfo(BaseModel):
    content: str
    condition: str
