import os
import pdfplumber
from bs4 import BeautifulSoup

def load_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        return open(filepath, "r", encoding="utf-8").read()
    elif ext == ".pdf":
        return extract_pdf_text(filepath)
    elif ext in [".html", ".htm"]:
        return extract_html_text(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_html_text(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n", strip=True)

def split_text(text, chunk_size=256, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def load_and_chunk_all(path_list):
    all_chunks = []
    for path in path_list:
        raw_text = load_text_from_file(path)
        chunks = split_text(raw_text)
        all_chunks.extend(chunks)
    return all_chunks