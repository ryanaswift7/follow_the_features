from pydantic import BaseModel


class SuiteIDs(BaseModel):
    prune: str
    sft: str
    eval: str
    experiment: str
