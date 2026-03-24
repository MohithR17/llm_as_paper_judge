
import os
import json
from pathlib import Path
from typing import List
from dataclasses import dataclass
import fitz  # PyMuPDF
# Review base models for all venues
@dataclass
class ACLReviewBaseModel:
    Summary: str
    Strengths: List[str]
    Weaknesses: List[str]
    Questions: List[str]
    Decision: int

@dataclass
class ICLRReviewBaseModel:
    Summary: str
    Strengths: List[str]
    Weaknesses: List[str]
    Questions: List[str]
    Decision: int

@dataclass
class CONLLReviewBaseModel:
    Summary: str
    Strengths: List[str]
    Weaknesses: List[str]
    Questions: List[str]
    Decision: int

@dataclass
class ArXivReviewBaseModel:
    Summary: str
    Strengths: List[str]
    Weaknesses: List[str]
    Questions: List[str]
    Decision: int

# 1. Document Ingestion: PDF to text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 2. Context Construction
def build_prompt(title, parsed_text, standard_rubric, venue="ICLR"):
    return f"""
### SYSTEM ROLE
You are an expert senior reviewer for {venue}. 
Your goal is to provide a critical, objective, and constructive review.

### INPUT DATA
Paper Title: {title}
Paper Content: {parsed_text}
Review Rubric: {standard_rubric}

### TASK
Provide a review following this structure:
1. Summary: What is the main contribution?
2. Strengths: List at least 3 technical strengths.
3. Weaknesses: Identify gaps in theory, experiments, or clarity.
4. Questions: Specific technical questions for the authors.
5. Decision: Score (1-10) and Recommendation (Accept/Reject).

### CONSTRAINT
Base your judgment only on the provided text. Do not use external tools.
"""

# 3. Example Standard Rubric (ICLR/NeurIPS)
STANDARD_RUBRIC = """
- Originality: Is the paper novel and creative?
- Soundness: Are the methods and conclusions technically correct?
- Clarity: Is the paper clearly written and well organized?
- Significance: Is the work important and impactful?
- Reproducibility: Are experiments and results reproducible?
"""

def load_paper_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    title = data.get("title", "Unknown Title")
    abstract = data.get("abstract", "")
    introduction = data.get("introduction", "")
    # Fallback: concatenate abstract and introduction
    parsed_text = abstract + "\n" + introduction
    return title, parsed_text

if __name__ == "__main__":
    # Example: process 1 paper from PeerRead ACL 2017 test split
    base = Path("PeerRead/data/acl_2017/test/")
    paper_json = base / "parsed_pdfs/323.pdf.json"
    title, parsed_text = load_paper_json(paper_json)
    prompt = build_prompt(title, parsed_text, STANDARD_RUBRIC, venue="ACL")
    print(prompt)
    # To call an LLM API, insert your API call here (e.g., OpenAI, Anthropic, etc.)
    # response = call_llm_api(prompt)
    # print(response)
