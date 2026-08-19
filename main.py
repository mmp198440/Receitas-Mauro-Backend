import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document
from google import genai
from google.genai import types

app = FastAPI(title="Receitas Mauro AI")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

class URLBody(BaseModel):
    url: str

class TextBody(BaseModel):
    text: str

SCHEMA_HINT = """
Devolve APENAS JSON válido com esta estrutura:
{
  "id": "uuid",
  "title": "Nome da receita em português",
  "category": "categoria curta em português",
  "favorite": false,
  "original_servings": 4,
  "prep_minutes": 0,
  "source_url": "",
  "source_type": "",
  "local_source_path": "",
  "ingredients": [
    {"name":"ingrediente em português","amount":100.0,"unit":"g"}
  ],
  "steps": ["passo 1 em português", "passo 2 em português"],
  "nutrition": {
    "kcal": 0.0,
    "protein": 0.0,
    "carbs": 0.0,
    "fat": 0.0,
    "fiber": 0.0
  },
  "notes": "",
  "created_at": "ISO8601"
}

REGRAS OBRIGATÓRIAS:
- A resposta final deve ficar SEMPRE em português europeu (pt-PT), independentemente do idioma da fonte.
- Traduz título, ingredientes, preparação, categoria e notas para português.
- Mantém marcas e nomes próprios sem tradução desnecessária.
- Extrai as quantidades com o máximo rigor possível.
- Se a receita indicar porções/pessoas, usa esse valor. Caso contrário estima sensatamente.
- A nutrição é POR PESSOA/PORÇÃO: kcal, proteína, hidratos, gordura e fibra.
- Se não houver valores nutricionais explícitos, estima-os com base nos ingredientes e quantidades.
- Não inventes ingredientes que não estejam na receita, exceto pequenas quantidades implícitas de água/sal quando necessário e assinala isso nas notas.
- Usa unidades práticas: g, kg, ml, L, unidade, colher de sopa, colher de chá, chávena.
"""

def clean_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    data = json.loads(text)
    data["id"] = data.get("id") or str(uuid.uuid4())
    data["created_at"] = data.get("created_at") or datetime.now().isoformat()
    return data

def call_recipe_ai(text: str, source_url: str = "", source_type: str = "", images: list[Path] | None = None):
    parts = [types.Part.from_text(text=f"Transforma a informação abaixo numa receita estruturada.\n{SCHEMA_HINT}\n\nFONTE:\n{text}\n\nURL original: {source_url}\nTipo de fonte: {source_type}")]
    for image in images or []:
        mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
        parts.append(types.Part.from_bytes(data=image.read_bytes(), mime_type=mime))
    r = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = clean_json(r.text or "{}")
    data["source_url"] = source_url or data.get("source_url", "")
    data["source_type"] = source_type or data.get("source_type", "")
    return data

def transcribe_audio(path: Path) -> str:
    # Envia o áudio diretamente para o Gemini; evita serviços de transcrição pagos separados.
    mime = mimetypes.guess_type(path.name)[0] or "audio/mpeg"
    r = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[
            types.Part.from_text(text="Transcreve fielmente o áudio deste vídeo. Mantém nomes de ingredientes, quantidades, tempos e temperaturas."),
            types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
        ])],
    )
    return r.text or ""

def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def video_context(video: Path, work: Path):
    audio = work / "audio.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vn", "-acodec", "libmp3lame", "-q:a", "4", str(audio)],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    transcript = transcribe_audio(audio) if audio.exists() and audio.stat().st_size > 0 else ""
    frames_dir = work / "frames"
    frames_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vf", "fps=1/8,scale=768:-1", "-frames:v", "10", str(frames_dir / "frame_%02d.jpg")],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return transcript, sorted(frames_dir.glob("*.jpg"))[:10]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/extract/text")
def extract_text(body: TextBody):
    if not body.text.strip():
        raise HTTPException(400, "Texto vazio")
    return call_recipe_ai(body.text, source_type="texto")

@app.post("/extract/url")
def extract_url(body: URLBody):
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL inválido")
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        template = str(work / "source.%(ext)s")
        result = subprocess.run(
            ["yt-dlp", "--no-playlist", "--write-description", "--write-info-json", "-o", template, body.url],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            return call_recipe_ai(
                f"Foi fornecido este link de uma receita/vídeo: {body.url}. Se não houver conteúdo suficiente, indica nas notas que o utilizador deve importar o vídeo diretamente.",
                source_url=body.url,
                source_type="link",
            )

        info_text = ""
        for p in work.iterdir():
            if p.suffix == ".description":
                info_text += "\nDESCRIÇÃO:\n" + p.read_text(errors="ignore")
            if p.name.endswith(".info.json"):
                try:
                    info = json.loads(p.read_text(errors="ignore"))
                    info_text += "\nTÍTULO:\n" + str(info.get("title", ""))
                    info_text += "\nDESCRIÇÃO METADATA:\n" + str(info.get("description", ""))
                except Exception:
                    pass

        videos = [p for p in work.iterdir() if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov", ".m4v"}]
        if videos:
            transcript, frames = video_context(videos[0], work)
            return call_recipe_ai(
                f"{info_text}\nTRANSCRIÇÃO DO ÁUDIO:\n{transcript}",
                source_url=body.url,
                source_type="vídeo online",
                images=frames,
            )
        return call_recipe_ai(info_text or body.url, source_url=body.url, source_type="link")

@app.post("/extract/file")
async def extract_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        path = work / ("input" + suffix)
        with path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        if suffix == ".txt":
            return call_recipe_ai(path.read_text(errors="ignore"), source_type="TXT")
        if suffix == ".pdf":
            return call_recipe_ai(extract_pdf(path), source_type="PDF")
        if suffix == ".docx":
            return call_recipe_ai(extract_docx(path), source_type="DOCX")
        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            return call_recipe_ai("Analisa esta imagem e extrai a receita visível.", source_type="imagem", images=[path])
        if suffix in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}:
            transcript, frames = video_context(path, work)
            return call_recipe_ai(f"TRANSCRIÇÃO DO ÁUDIO:\n{transcript}", source_type="vídeo importado", images=frames)
        raise HTTPException(400, f"Formato não suportado: {suffix}")
