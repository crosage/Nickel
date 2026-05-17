"""
Paper Reader — FastAPI 后端
===========================
功能:
  1. 从 CVF Open Access 爬取论文列表（带缓存）
  2. SQLite 存储: 论文元数据、阅读状态、用户笔记
  3. 下载 PDF → 智能提取关键 section → LLM 解读
  4. RESTful API 供 Flutter 前端调用

启动:
    pip install fastapi uvicorn requests beautifulsoup4 pypdf
    python server.py
    # 或
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import re
import time
import json
import sqlite3
import hashlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from contextlib import contextmanager
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

CVF_BASE_URL = "https://openaccess.thecvf.com"
SERVER_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PAPER_READER_DATA_DIR", SERVER_DIR / "paper_reader_data")).expanduser()
PDF_DIR = DATA_DIR / "pdfs"
PAGE_IMAGE_DIR = DATA_DIR / "page_images"
FIGURE_DIR = DATA_DIR / "figures"
DB_PATH = DATA_DIR / "papers.db"

DEFAULT_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
DEFAULT_LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
DEFAULT_LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "420"))
AUTO_TRANSLATE_ON_START = os.environ.get("AUTO_TRANSLATE_ON_START", "0").lower() not in ("0", "false", "no")
AUTO_TRANSLATE_LIMIT = int(os.environ.get("AUTO_TRANSLATE_LIMIT", "0"))
PDF_IMAGE_DPI = int(os.environ.get("PDF_IMAGE_DPI", "144"))
FIGURE_IMAGE_DPI = int(os.environ.get("FIGURE_IMAGE_DPI", "180"))
MAX_EXTRACTED_FIGURES = int(os.environ.get("MAX_EXTRACTED_FIGURES", "16"))

# Section 提取上限
SECTION_MAX_CHARS = {
    "abstract": 3000, "introduction": 8000,
    "method": 15000, "experiments": 8000, "conclusion": 4000,
}
TOTAL_MAX_CHARS = 40000
MIN_TRANSLATION_SOURCE_CHARS = 6000
TRANSLATION_CHUNK_MAX_CHARS = int(os.environ.get("TRANSLATION_CHUNK_MAX_CHARS", "9000"))
TRANSLATION_CHUNK_OVERLAP = int(os.environ.get("TRANSLATION_CHUNK_OVERLAP", "0"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRANSLATION_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper_translation")
TRANSLATION_JOBS: dict[str, Future] = {}
TRANSLATION_LOCK = threading.Lock()
AUTO_TRANSLATE_STARTED = False

# ─────────────────────────────────────────────
# 数据库
# ─────────────────────────────────────────────

def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS papers (
        id TEXT PRIMARY KEY,
        conference TEXT NOT NULL,
        title TEXT NOT NULL,
        authors TEXT DEFAULT '[]',
        abstract TEXT DEFAULT '',
        pdf_url TEXT DEFAULT '',
        page_url TEXT DEFAULT '',
        arxiv_url TEXT DEFAULT '',
        pages TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS reading_status (
        paper_id TEXT PRIMARY KEY REFERENCES papers(id),
        status TEXT DEFAULT 'unread',  -- unread / reading / read
        progress REAL DEFAULT 0.0,
        updated_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id TEXT NOT NULL REFERENCES papers(id),
        content TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS analyses (
        paper_id TEXT PRIMARY KEY REFERENCES papers(id),
        sections_json TEXT DEFAULT '{}',
        analysis TEXT DEFAULT '',
        model TEXT DEFAULT '',
        token_count INTEGER DEFAULT 0,
        translation TEXT DEFAULT '',
        translation_model TEXT DEFAULT '',
        translation_token_count INTEGER DEFAULT 0,
        translation_created_at TEXT DEFAULT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS translation_chunks (
        paper_id TEXT NOT NULL REFERENCES papers(id),
        chunk_index INTEGER NOT NULL,
        chunk_total INTEGER NOT NULL,
        source_hash TEXT NOT NULL,
        source_label TEXT DEFAULT '',
        source_text TEXT DEFAULT '',
        translation TEXT DEFAULT '',
        model TEXT DEFAULT '',
        token_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        error TEXT DEFAULT '',
        updated_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (paper_id, chunk_index)
    );

    CREATE TABLE IF NOT EXISTS figure_refs (
        paper_id TEXT NOT NULL REFERENCES papers(id),
        figure_no TEXT NOT NULL,
        url TEXT NOT NULL,
        caption TEXT DEFAULT '',
        page INTEGER DEFAULT 0,
        first_ref_offset INTEGER DEFAULT -1,
        first_ref_text TEXT DEFAULT '',
        confidence REAL DEFAULT 0.0,
        updated_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (paper_id, figure_no)
    );

    CREATE TABLE IF NOT EXISTS sync_log (
        conference TEXT PRIMARY KEY,
        last_sync TEXT DEFAULT (datetime('now')),
        paper_count INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_papers_conf ON papers(conference);
    CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);
    CREATE INDEX IF NOT EXISTS idx_reading_status ON reading_status(status);
    CREATE INDEX IF NOT EXISTS idx_notes_paper ON notes(paper_id);
    CREATE INDEX IF NOT EXISTS idx_translation_chunks_paper ON translation_chunks(paper_id, status);
    CREATE INDEX IF NOT EXISTS idx_figure_refs_paper ON figure_refs(paper_id);
    """)
    for ddl in [
        "ALTER TABLE analyses ADD COLUMN translation TEXT DEFAULT ''",
        "ALTER TABLE analyses ADD COLUMN translation_model TEXT DEFAULT ''",
        "ALTER TABLE analyses ADD COLUMN translation_token_count INTEGER DEFAULT 0",
        "ALTER TABLE analyses ADD COLUMN translation_created_at TEXT DEFAULT NULL",
    ]:
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError as e:
            if 'duplicate column name' not in str(e).lower():
                raise
    conn.close()


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────
# CVF 爬取 (带缓存)
# ─────────────────────────────────────────────

def sync_conference(conference: str, day: Optional[str] = None, force: bool = False):
    """从 CVF 同步论文列表到数据库。"""
    with get_db() as db:
        if not force:
            row = db.execute(
                "SELECT last_sync, paper_count FROM sync_log WHERE conference=?",
                (conference,)
            ).fetchone()
            if row:
                last = datetime.fromisoformat(row["last_sync"])
                age_hours = (datetime.now() - last).total_seconds() / 3600
                if age_hours < 24:
                    log.info(f"{conference} 已同步 ({row['paper_count']} 篇, {age_hours:.1f}h 前)")
                    return row["paper_count"]

    # 爬取
    if day:
        url = f"{CVF_BASE_URL}/{conference}?day={day}"
    else:
        url = f"{CVF_BASE_URL}/{conference}?day=all"

    log.info(f"同步 {conference}: {url}")
    headers = {"User-Agent": "PaperReader/1.0 (Academic)"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # 解析论文列表
    papers = []
    dt_list = soup.find_all("dt", class_="ptitle")

    if dt_list:
        for dt in dt_list:
            a_tag = dt.find("a")
            if not a_tag:
                continue
            href = a_tag.get("href", "")
            title = a_tag.get_text(strip=True)
            if not title or not href:
                continue

            page_url = urljoin(CVF_BASE_URL, href)
            paper_id = hashlib.md5(title.encode()).hexdigest()[:16]
            pdf_url = page_url.replace("/html/", "/papers/").replace("_paper.html", "_paper.pdf")

            dd = dt.find_next_sibling("dd")
            authors, pages_str = [], ""
            if dd:
                dd_text = dd.get_text(strip=True)
                if ";" in dd_text:
                    authors = [a.strip() for a in dd_text.split(";")[0].split(",") if a.strip()]
                pm = re.search(r"pp\.\s*(\d+-\d+)", dd_text)
                if pm:
                    pages_str = pm.group(0)

            papers.append({
                "id": paper_id,
                "conference": conference,
                "title": title,
                "authors": json.dumps(authors, ensure_ascii=False),
                "pdf_url": pdf_url,
                "page_url": page_url,
                "pages": pages_str,
            })
    else:
        # 备选解析
        links = soup.find_all("a", href=re.compile(r"/content/.*_paper\.html$"))
        for a in links:
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if not title or not href:
                continue
            page_url = urljoin(CVF_BASE_URL, href)
            paper_id = hashlib.md5(title.encode()).hexdigest()[:16]
            pdf_url = page_url.replace("/html/", "/papers/").replace("_paper.html", "_paper.pdf")
            papers.append({
                "id": paper_id,
                "conference": conference,
                "title": title,
                "authors": "[]",
                "pdf_url": pdf_url,
                "page_url": page_url,
            })

    # 写入数据库
    with get_db() as db:
        for p in papers:
            db.execute("""
                INSERT OR IGNORE INTO papers (id, conference, title, authors, pdf_url, page_url, pages)
                VALUES (:id, :conference, :title, :authors, :pdf_url, :page_url, :pages)
            """, p)
        db.execute("""
            INSERT OR REPLACE INTO sync_log (conference, last_sync, paper_count)
            VALUES (?, datetime('now'), ?)
        """, (conference, len(papers)))

    log.info(f"同步完成: {len(papers)} 篇")
    return len(papers)


def fetch_paper_abstract(paper_id: str):
    """从 CVF 详情页获取 abstract。"""
    with get_db() as db:
        row = db.execute("SELECT page_url, abstract FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not row:
            return None
        if row["abstract"]:
            return row["abstract"]

    page_url = row["page_url"]
    try:
        resp = requests.get(page_url, timeout=20, headers={"User-Agent": "PaperReader/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        abstract = ""
        # 尝试多种方式
        for tag in soup.find_all(["strong", "b", "h3", "div"]):
            if "Abstract" in tag.get_text():
                next_el = tag.find_next(["p", "div"])
                if next_el and len(next_el.get_text(strip=True)) > 50:
                    abstract = next_el.get_text(strip=True)
                    break

        if not abstract:
            page_text = soup.get_text()
            m = re.search(r"Abstract\s*\n+(.*?)(?:\n\s*\n|Related Material)", page_text, re.DOTALL)
            if m:
                abstract = m.group(1).strip()

        # 同时尝试获取 arxiv 链接
        arxiv_url = ""
        for a in soup.find_all("a", href=True):
            if "arxiv.org" in a.get("href", ""):
                arxiv_url = a["href"]
                break

        with get_db() as db:
            db.execute("UPDATE papers SET abstract=?, arxiv_url=? WHERE id=?",
                       (abstract, arxiv_url, paper_id))

        return abstract
    except Exception as e:
        log.warning(f"获取 abstract 失败: {e}")
        return ""


# ─────────────────────────────────────────────
# PDF 下载 + Section 提取
# ─────────────────────────────────────────────

SECTION_PATTERNS = {
    "abstract": [re.compile(r"^(?:Abstract|ABSTRACT)\s*$", re.MULTILINE)],
    "introduction": [re.compile(r"^(?:\d+\.?\s*)?(?:Introduction|INTRODUCTION)\s*$", re.MULTILINE)],
    "method": [re.compile(
        r"^(?:\d+\.?\s*)?(?:Method(?:ology|s)?|Approach|Proposed\s+(?:Method|Approach|Framework)|"
        r"Our\s+(?:Method|Approach)|Framework|Model|METHODS?)\s*$",
        re.MULTILINE | re.IGNORECASE)],
    "experiments": [re.compile(
        r"^(?:\d+\.?\s*)?(?:Experiments?|Results?|Experimental|Evaluation|EXPERIMENTS?)\s*$",
        re.MULTILINE | re.IGNORECASE)],
    "conclusion": [re.compile(
        r"^(?:\d+\.?\s*)?(?:Conclusion|Conclusions|CONCLUSION|Summary|Concluding\s+Remarks)\s*$",
        re.MULTILINE | re.IGNORECASE)],
}

NEXT_SECTION_RE = re.compile(
    r"^(?:\d+\.?\s+)?(?:Introduction|Related|Background|Preliminaries|"
    r"Method|Approach|Proposed|Our\s+|Framework|"
    r"Experiment|Results|Evaluation|Ablation|"
    r"Conclusion|Summary|Discussion|Acknowledg|References|REFERENCES|Appendix)\b",
    re.MULTILINE | re.IGNORECASE,
)


def ensure_pdf_downloaded(paper_id: str) -> Path:
    """Return local PDF path, downloading it if needed."""
    with get_db() as db:
        paper = db.execute("SELECT pdf_url FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not paper:
            raise HTTPException(404, "paper not found")

    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return pdf_path

    try:
        log.info(f"下载 PDF: {paper['pdf_url']}")
        resp = requests.get(
            paper["pdf_url"],
            timeout=60,
            stream=True,
            headers={"User-Agent": "PaperReader/1.0"},
        )
        resp.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
    except Exception as e:
        log.error(f"下载失败: {e}")
        raise HTTPException(502, f"PDF download failed: {e}")

    return pdf_path


def render_pdf_page_images(paper_id: str, max_pages: int = 6) -> list[Path]:
    """Render leading PDF pages to JPEG files for frontend viewing and VLM input."""
    pdf_path = ensure_pdf_downloaded(paper_id)
    out_dir = PAGE_IMAGE_DIR / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("page_*.jpg"))
    if len(existing) >= max_pages:
        return existing[:max_pages]

    try:
        import fitz
    except Exception as e:
        raise HTTPException(500, f"PyMuPDF is required for page images: {e}")

    try:
        doc = fitz.open(str(pdf_path))
        limit = min(max_pages, len(doc))
        zoom = PDF_IMAGE_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        paths = []
        for idx in range(limit):
            image_path = out_dir / f"page_{idx + 1:03d}.jpg"
            if not image_path.exists():
                page = doc.load_page(idx)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pix.save(str(image_path), output="jpeg", jpg_quality=82)
            paths.append(image_path)
        doc.close()
        return paths
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"PDF page rendering failed: {e}")
        raise HTTPException(500, f"PDF page rendering failed: {e}")


FIGURE_CAPTION_RE = re.compile(r"(?i)\b(?:fig(?:ure)?\.?)\s*([0-9]+)\b")


def extract_pdf_figures(paper_id: str, max_figures: int = MAX_EXTRACTED_FIGURES) -> list[dict]:
    """Crop original figure regions near Fig./Figure captions for reinsertion in translations."""
    out_dir = FIGURE_DIR / paper_id
    meta_path = out_dir / "figures.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))[:max_figures]
        except Exception:
            pass

    pdf_path = ensure_pdf_downloaded(paper_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import fitz
    except Exception as e:
        raise HTTPException(500, f"PyMuPDF is required for figure extraction: {e}")

    figures = []
    seen_numbers = set()
    try:
        doc = fitz.open(str(pdf_path))
        zoom = FIGURE_IMAGE_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(len(doc)):
            if len(figures) >= max_figures:
                break
            page = doc.load_page(page_index)
            blocks = page.get_text("blocks")
            page_rect = page.rect
            for block in blocks:
                if len(figures) >= max_figures:
                    break
                text = (block[4] or "").strip()
                match = FIGURE_CAPTION_RE.search(text)
                if not match:
                    continue
                fig_no = match.group(1)
                if fig_no in seen_numbers:
                    continue
                seen_numbers.add(fig_no)

                caption_rect = fitz.Rect(block[:4])
                clip = fitz.Rect(
                    page_rect.x0,
                    max(page_rect.y0, caption_rect.y0 - 360),
                    page_rect.x1,
                    min(page_rect.y1, caption_rect.y1 + 80),
                )
                if clip.height < 120:
                    continue

                image_path = out_dir / f"figure_{fig_no}.jpg"
                pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
                pix.save(str(image_path), output="jpeg", jpg_quality=88)
                figures.append({
                    "number": fig_no,
                    "label": f"Figure {fig_no}",
                    "caption": re.sub(r"\s+", " ", text)[:500],
                    "page": page_index + 1,
                    "url": f"/api/papers/{paper_id}/figures/{fig_no}",
                })
        doc.close()
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"PDF figure extraction failed: {e}")
        raise HTTPException(500, f"PDF figure extraction failed: {e}")

    meta_path.write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
    return figures[:max_figures]


def build_figure_reference_map(paper_id: str, translation: str) -> list[dict]:
    """Build structured figure insertion metadata from translated reference positions."""
    try:
        figures = extract_pdf_figures(paper_id)
    except Exception as e:
        log.warning(f"Figure map build skipped: {e}")
        return []
    refs = []
    for fig in figures:
        no = str(fig.get("number", "")).strip()
        if not no:
            continue
        pattern = re.compile(
            rf"(?P<ref>(?:图|Figure|Fig\.?)\s*{re.escape(no)}(?:[^\n。；;]*[。；;]?)?)",
            re.IGNORECASE,
        )
        match = pattern.search(translation)
        offset = match.start() if match else -1
        ref_text = match.group("ref")[:300] if match else ""
        confidence = 0.9 if match else 0.35
        refs.append({
            "paper_id": paper_id,
            "figure_no": no,
            "url": fig.get("url", ""),
            "caption": fig.get("caption", ""),
            "page": int(fig.get("page") or 0),
            "first_ref_offset": offset,
            "first_ref_text": ref_text,
            "confidence": confidence,
        })
    refs.sort(key=lambda r: (r["first_ref_offset"] < 0, r["first_ref_offset"], int(r["figure_no"]) if r["figure_no"].isdigit() else 9999))
    with get_db() as db:
        db.execute("DELETE FROM figure_refs WHERE paper_id=?", (paper_id,))
        for ref in refs:
            db.execute("""
                INSERT INTO figure_refs (
                    paper_id, figure_no, url, caption, page,
                    first_ref_offset, first_ref_text, confidence, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(paper_id, figure_no) DO UPDATE SET
                    url=excluded.url,
                    caption=excluded.caption,
                    page=excluded.page,
                    first_ref_offset=excluded.first_ref_offset,
                    first_ref_text=excluded.first_ref_text,
                    confidence=excluded.confidence,
                    updated_at=datetime('now')
            """, (
                ref["paper_id"], ref["figure_no"], ref["url"], ref["caption"],
                ref["page"], ref["first_ref_offset"], ref["first_ref_text"], ref["confidence"],
            ))
    return refs


def insert_figure_markdown_by_refs(translation: str, paper_id: str) -> str:
    """Insert figure markdown after the translated paragraph containing first semantic reference."""
    if not translation.strip():
        return translation
    if f"/api/papers/{paper_id}/figures/" in translation:
        return translation

    refs = build_figure_reference_map(paper_id, translation)
    result = translation
    shift = 0
    for ref in refs:
        offset = int(ref.get("first_ref_offset", -1))
        if offset < 0:
            continue
        url = ref.get("url", "")
        no = ref.get("figure_no", "")
        if not url or not no:
            continue
        marker = f"![Figure {no}]({url})"
        insert_from = offset + shift
        insert_at = result.find("\n\n", insert_from)
        if insert_at == -1:
            insert_at = len(result)
            insertion = f"\n\n{marker}\n"
        else:
            insertion = f"\n\n{marker}"
        result = result[:insert_at] + insertion + result[insert_at:]
        shift += len(insertion)
    return result


def download_and_extract(paper_id: str) -> dict:
    """下载 PDF，提取关键 section，返回 sections dict。"""
    with get_db() as db:
        # 检查缓存
        cached = db.execute("SELECT sections_json FROM analyses WHERE paper_id=?", (paper_id,)).fetchone()
        if cached and cached["sections_json"] and cached["sections_json"] != "{}":
            return json.loads(cached["sections_json"])

        paper = db.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not paper:
            return {}

    try:
        pdf_path = ensure_pdf_downloaded(paper_id)
    except HTTPException:
        return {}

    # 提取文本
    try:
        reader = PdfReader(str(pdf_path))
        full_text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    except Exception as e:
        log.error(f"PDF 解析失败: {e}")
        return {}

    # 提取 sections
    sections = {}
    for sec_name, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            match = pat.search(full_text)
            if not match:
                continue
            start = match.end()
            remaining = full_text[start:]
            end_pos = len(remaining)
            for nm in NEXT_SECTION_RE.finditer(remaining):
                if nm.start() > 50:
                    end_pos = nm.start()
                    break
            text = remaining[:end_pos].strip()
            max_c = SECTION_MAX_CHARS.get(sec_name, 8000)
            if len(text) > max_c:
                text = text[:max_c] + "\n[... 截断 ...]"
            if text:
                sections[sec_name] = text
            break

    # 缓存。不要用 INSERT OR REPLACE，否则会把已有 analysis/translation 列覆盖为空。
    with get_db() as db:
        db.execute("""
            INSERT INTO analyses (paper_id, sections_json)
            VALUES (?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET sections_json=excluded.sections_json
        """, (paper_id, json.dumps(sections, ensure_ascii=False)))

    return sections


def extract_full_text_for_translation(paper_id: str) -> str:
    """Extract broad PDF text as a fallback when section detection is too sparse."""
    with get_db() as db:
        paper = db.execute("SELECT pdf_url FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not paper:
            return ""

    try:
        pdf_path = ensure_pdf_downloaded(paper_id)
    except HTTPException as e:
        log.error(f"Fallback PDF download failed: {e.detail}")
        return ""

    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for i, page in enumerate(reader.pages):
            if i >= 20:
                break
            pages.append(page.extract_text() or "")
        full_text = "\n\n".join(pages)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        full_text = re.sub(r"(?i)\breferences\b[\s\S]*$", "", full_text)
        return full_text.strip()
    except Exception as e:
        log.error(f"Fallback PDF extraction failed: {e}")
        return ""


def sections_need_full_text_fallback(sections: dict) -> bool:
    total_chars = sum(len(v or "") for v in sections.values())
    has_body = bool(sections.get("method") or sections.get("experiments"))
    return total_chars < MIN_TRANSLATION_SOURCE_CHARS or not has_body


# ─────────────────────────────────────────────
# LLM 解读
# ─────────────────────────────────────────────

ANALYSIS_PROMPT = """你是一位顶尖的计算机视觉（CV）算法专家和顶会审稿人。请基于提供的论文关键章节，输出结构化、极度硬核的中文解读。

请严格按照以下 6 个部分输出，拒绝假大空的废话，必须直击技术细节：

### 1. 一句话总结
- 用一句话高度凝练：本文针对什么具体的 CV 任务（如语义分割、目标检测等），提出了什么核心结构/机制，最终在什么数据集上取得了怎样的突破或 SOTA 结果。

### 2. 研究背景与动机
- 现有算法的具体痛点是什么？（例如：长距离上下文依赖不足、边缘细节丢失、特征对齐困难、感受野受限等）
- 本文打破僵局的核心 Insight（直觉或切入点）是什么？

### 3. 核心方法（关键创新点 2-4 个）
*(这是重点，请详细拆解网络架构与机制)*
- **整体 Pipeline**：按顺序描述数据流向，输入张量的维度经历了怎样的变化？最终输出是什么？
- **模块拆解**：详细剖析其提出的 2-4 个关键模块。它是如何进行特征提取、融合与精细预测的？
- **损失函数**：使用了哪些具体的 Loss 进行联合优化？为什么要这样设计？

### 4. 实验与结果（关键数字）
- **验证环境**：在哪些公开数据集上进行了测试？
- **量化提升**：**必须提取具体的数字！** 对比主流 Baseline，核心指标（如 mIoU、AP、Dice 等）具体提升了多少？
- **效率指标**：如果有，请务必提取模型参数量（Params）、计算量（FLOPs）或推理速度（FPS）的量化数据。
- **消融实验**：哪个创新模块对最终精度的贡献最大？

### 5. 优势与局限
- **核心优势**：该方法在实际工程落地中的最大卖点是什么？
- **局限与算力瓶颈**：探讨其面对超高分辨率图像（如 4000x4000、8000x8000 级别的大尺度输入）时，该架构是否存在显存爆炸、推理延迟骤增或局部感受野失效的潜在风险？

### 6. 关键结论与启发
- 从这篇论文的架构设计中，能得出什么经得起推敲的核心结论？
- 对同领域的后续研究或实际算法开发有什么具体的启发？

清晰，用直觉语言解释复杂概念。如章节缺失请说明。
请务必详尽展开每个部分，总输出不少于 2000 字。每个模块的描述必须包含具体的技术术语、张量维度、公式含义或数值结果，禁止一笔带过。"""


TRANSLATION_CHUNK_PROMPT = """You are a rigorous academic paper translator.
Translate only the given paper segment into Simplified Chinese.

Requirements:
- Preserve section titles, paragraph order, formula numbers, figure/table numbers, citation numbers, proper nouns, abbreviations, and code links.
- Do not summarize, explain, expand, omit, or rewrite the segment.
- Do not add prefaces such as "Here is the translation".
- If the segment begins or ends mid-section, translate only the visible content.

Only output the translation result."""


def _build_llm_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith(("/v1", "/v2", "/v3")):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _call_llm(messages: list[dict], max_tokens: int, temperature: float = 0.3) -> dict:
    api_key = DEFAULT_LLM_API_KEY
    if not api_key:
        return {"error": "[LLM API key is not configured]"}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    url = _build_llm_url(DEFAULT_LLM_BASE_URL)
    resp = requests.post(url, headers=headers, timeout=DEFAULT_LLM_TIMEOUT, json={
        "model": DEFAULT_LLM_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    })
    resp.raise_for_status()
    return resp.json()


def split_translation_text(text: str, max_chars: int = TRANSLATION_CHUNK_MAX_CHARS) -> list[str]:
    """Split long paper text at paragraph boundaries to avoid LLM truncation."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary <= start + max_chars // 2:
                boundary = text.rfind("\n", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - TRANSLATION_CHUNK_OVERLAP, 0)
        if TRANSLATION_CHUNK_OVERLAP > 0:
            next_para = text.find("\n\n", start, min(start + TRANSLATION_CHUNK_OVERLAP + 200, len(text)))
            if next_para != -1 and next_para < end:
                start = next_para + 2
    return chunks


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def translate_single_chunk(title: str, authors: str, chunk: str, idx: int, total: int) -> tuple[str, int]:
    header = (
        f"Paper title: {title}\n"
        f"Authors: {authors}\n"
        f"Segment: {idx}/{total}\n\n"
        f"<paper_segment>\n{chunk}\n</paper_segment>"
    )
    data = _call_llm([
        {"role": "system", "content": TRANSLATION_CHUNK_PROMPT},
        {"role": "user", "content": header},
    ], max_tokens=10000, temperature=0.1)
    if "error" in data:
        raise RuntimeError(data["error"])
    translated = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return translated, data.get("usage", {}).get("total_tokens", 0)


def translate_text_chunks(title: str, authors: str, text: str) -> tuple[str, int]:
    chunks = split_translation_text(text)
    translated_chunks = []
    total_tokens = 0
    for idx, chunk in enumerate(chunks, start=1):
        translated, tokens = translate_single_chunk(title, authors, chunk, idx, len(chunks))
        translated_chunks.append(translated)
        total_tokens += tokens
    return "\n\n".join(translated_chunks), total_tokens


def translate_text_chunks_cached(paper_id: str, title: str, authors: str, text: str, force: bool = False) -> tuple[str, int]:
    chunks = split_translation_text(text)
    if not chunks:
        return "", 0

    total_tokens = 0
    translations = []
    with get_db() as db:
        if force:
            db.execute("DELETE FROM translation_chunks WHERE paper_id=?", (paper_id,))

    for idx, chunk in enumerate(chunks, start=1):
        h = source_hash(chunk)
        cached = None
        if not force:
            with get_db() as db:
                cached = db.execute("""
                    SELECT translation, token_count FROM translation_chunks
                    WHERE paper_id=? AND chunk_index=? AND chunk_total=?
                      AND source_hash=? AND status='done' AND translation!=''
                """, (paper_id, idx, len(chunks), h)).fetchone()
        if cached:
            translations.append(cached["translation"])
            total_tokens += cached["token_count"] or 0
            continue

        with get_db() as db:
            db.execute("""
                INSERT INTO translation_chunks (
                    paper_id, chunk_index, chunk_total, source_hash, source_label,
                    source_text, status, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'running', datetime('now'))
                ON CONFLICT(paper_id, chunk_index) DO UPDATE SET
                    chunk_total=excluded.chunk_total,
                    source_hash=excluded.source_hash,
                    source_label=excluded.source_label,
                    source_text=excluded.source_text,
                    status='running',
                    error='',
                    updated_at=datetime('now')
            """, (paper_id, idx, len(chunks), h, f"Segment {idx}/{len(chunks)}", chunk))
        try:
            translated, tokens = translate_single_chunk(title, authors, chunk, idx, len(chunks))
            with get_db() as db:
                db.execute("""
                    UPDATE translation_chunks
                    SET translation=?, model=?, token_count=?, status='done',
                        error='', updated_at=datetime('now')
                    WHERE paper_id=? AND chunk_index=?
                """, (translated, DEFAULT_LLM_MODEL, tokens, paper_id, idx))
            translations.append(translated)
            total_tokens += tokens
        except Exception as e:
            with get_db() as db:
                db.execute("""
                    UPDATE translation_chunks
                    SET status='failed', error=?, updated_at=datetime('now')
                    WHERE paper_id=? AND chunk_index=?
                """, (str(e), paper_id, idx))
            raise

    return "\n\n".join(translations), total_tokens


def analyze_paper(paper_id: str) -> str:
    """调用 LLM 解读论文。"""
    with get_db() as db:
        cached = db.execute("SELECT analysis FROM analyses WHERE paper_id=?", (paper_id,)).fetchone()
        if cached and cached["analysis"]:
            return cached["analysis"]

        paper = db.execute("SELECT title, authors, abstract FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not paper:
            return ""

    sections = download_and_extract(paper_id)
    if not sections:
        return "[无法提取论文内容]"

    # 构建 prompt
    parts = [f"标题: {paper['title']}", f"作者: {paper['authors']}", ""]
    labels = {"abstract": "Abstract", "introduction": "Introduction",
              "method": "Method", "experiments": "Experiments", "conclusion": "Conclusion"}
    for key in ["abstract", "introduction", "method", "experiments", "conclusion"]:
        if key in sections:
            parts.append(f"=== {labels[key]} ===")
            parts.append(sections[key])
            parts.append("")

    if sections_need_full_text_fallback(sections):
        fallback = extract_full_text_for_translation(paper_id)
        if len(fallback) > sum(len(v or "") for v in sections.values()):
            log.info(f"Using full-text fallback for translation: {paper_id} | {len(fallback)} chars")
            parts = [
                f"Title: {paper['title']}",
                f"Authors: {paper['authors']}",
                "",
                "=== Full Paper Text Fallback ===",
                fallback,
                "",
            ]

    text = "\n".join(parts)
    if len(text) > TOTAL_MAX_CHARS:
        text = text[:TOTAL_MAX_CHARS] + "\n[截断]"

    try:
        data = _call_llm([
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user", "content": "Please analyze:\n\n<paper>\n" + text + "\n</paper>"},
        ], max_tokens=10240, temperature=0.3)
        if "error" in data:
            return data["error"]
        analysis = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        tokens = data.get("usage", {}).get("total_tokens", 0)
        with get_db() as db:
            db.execute("""
                INSERT INTO analyses (paper_id, analysis, model, token_count, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(paper_id) DO UPDATE SET
                    analysis=excluded.analysis,
                    model=excluded.model,
                    token_count=excluded.token_count,
                    created_at=datetime('now')
            """, (paper_id, analysis, DEFAULT_LLM_MODEL, tokens))

        return analysis
    except Exception as e:
        log.error(f"LLM analysis error: {e}")
        return f"[LLM analysis failed: {e}]"



def translate_paper(paper_id: str) -> str:
    """Call the LLM to generate a Chinese translation for the paper."""
    with get_db() as db:
        cached = db.execute("SELECT translation FROM analyses WHERE paper_id=?", (paper_id,)).fetchone()
        if cached and cached["translation"]:
            translation = cached["translation"]
            if f"/api/papers/{paper_id}/figures/" not in translation:
                enriched_translation = insert_figure_markdown_by_refs(translation, paper_id)
                if enriched_translation != translation:
                    db.execute("UPDATE analyses SET translation=? WHERE paper_id=?", (enriched_translation, paper_id))
                    translation = enriched_translation
            return translation

        paper = db.execute("SELECT title, authors FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not paper:
            return ""

    sections = download_and_extract(paper_id)
    if not sections:
        return "[Unable to extract paper content]"

    parts = [f"Title: {paper['title']}", f"Authors: {paper['authors']}", ""]
    labels = {"abstract": "Abstract", "introduction": "Introduction",
              "method": "Method", "experiments": "Experiments", "conclusion": "Conclusion"}
    for key in ["abstract", "introduction", "method", "experiments", "conclusion"]:
        if key in sections:
            parts.append(f"=== {labels[key]} ===")
            parts.append(sections[key])
            parts.append("")

    if sections_need_full_text_fallback(sections):
        fallback = extract_full_text_for_translation(paper_id)
        if len(fallback) > sum(len(v or "") for v in sections.values()):
            log.info(f"Using full-text fallback for translation: {paper_id} | {len(fallback)} chars")
            parts = [
                f"Title: {paper['title']}",
                f"Authors: {paper['authors']}",
                "",
                "=== Full Paper Text Fallback ===",
                fallback,
                "",
            ]

    text = "\n".join(parts)

    try:
        translation, tokens = translate_text_chunks_cached(paper_id, paper["title"], paper["authors"], text)
        translation = insert_figure_markdown_by_refs(translation, paper_id)

        with get_db() as db:
            db.execute("""
                INSERT INTO analyses (
                    paper_id, translation, translation_model,
                    translation_token_count, translation_created_at
                )
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(paper_id) DO UPDATE SET
                    translation=excluded.translation,
                    translation_model=excluded.translation_model,
                    translation_token_count=excluded.translation_token_count,
                    translation_created_at=datetime('now')
            """, (paper_id, translation, DEFAULT_LLM_MODEL, tokens))

        return translation
    except Exception as e:
        log.error(f"LLM translation error: {e}")
        return f"[LLM translation failed: {e}]"


def _translation_job_status(paper_id: str) -> str:
    with TRANSLATION_LOCK:
        future = TRANSLATION_JOBS.get(paper_id)
        if not future:
            return ""
        if future.running():
            return "running"
        if not future.done():
            return "queued"
        err = future.exception()
        return "failed" if err else "done"


def _run_translation_job(paper_id: str) -> None:
    log.info(f"Background translation started: {paper_id}")
    translation = translate_paper(paper_id)
    if translation.startswith("["):
        raise RuntimeError(translation)
    log.info(f"Background translation finished: {paper_id} | {len(translation)} chars")


def queue_translation(paper_id: str, force: bool = False) -> dict:
    if not DEFAULT_LLM_API_KEY:
        raise HTTPException(400, "LLM_API_KEY is not configured")

    with get_db() as db:
        paper_exists = db.execute("SELECT 1 FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not paper_exists:
            raise HTTPException(404, "paper not found")
        row = db.execute("SELECT translation FROM analyses WHERE paper_id=?", (paper_id,)).fetchone()
        if row and row["translation"] and not force:
            return {
                "paper_id": paper_id,
                "status": "cached",
                "cached": True,
                "translation_length": len(row["translation"]),
                "translation": row["translation"],
            }
        if force:
            db.execute("""
                UPDATE analyses
                SET translation='', translation_model='', translation_token_count=0, translation_created_at=NULL
                WHERE paper_id=?
            """, (paper_id,))

    with TRANSLATION_LOCK:
        future = TRANSLATION_JOBS.get(paper_id)
        if future and not future.done():
            return {
                "paper_id": paper_id,
                "status": "running" if future.running() else "queued",
                "cached": False,
                "queued": True,
            }
        future = TRANSLATION_EXECUTOR.submit(_run_translation_job, paper_id)
        TRANSLATION_JOBS[paper_id] = future
        return {"paper_id": paper_id, "status": "queued", "cached": False, "queued": True}


def list_untranslated_papers(limit: int = 0) -> list[str]:
    sql = """
        SELECT p.id
        FROM papers p
        LEFT JOIN analyses a ON p.id = a.paper_id
        LEFT JOIN reading_status rs ON p.id = rs.paper_id
        WHERE a.translation IS NULL OR a.translation = ''
        ORDER BY
            CASE rs.status WHEN 'reading' THEN 0 WHEN 'read' THEN 1 ELSE 2 END,
            p.created_at DESC,
            p.id
    """
    params = ()
    if limit > 0:
        sql += " LIMIT ?"
        params = (limit,)
    with get_db() as db:
        return [row["id"] for row in db.execute(sql, params).fetchall()]


def queue_missing_translations(limit: int = 0) -> dict:
    if not DEFAULT_LLM_API_KEY:
        return {
            "requested": 0,
            "queued": 0,
            "cached": 0,
            "already_running": 0,
            "limit": limit,
            "error": "LLM_API_KEY is not configured",
        }

    paper_ids = list_untranslated_papers(limit)
    queued = 0
    cached = 0
    already_running = 0
    for paper_id in paper_ids:
        result = queue_translation(paper_id)
        status = result.get("status")
        if status == "cached":
            cached += 1
        elif status in ("queued",):
            queued += 1
        elif status == "running":
            already_running += 1
    return {
        "requested": len(paper_ids),
        "queued": queued,
        "cached": cached,
        "already_running": already_running,
        "limit": limit,
    }


def translation_queue_summary() -> dict:
    with TRANSLATION_LOCK:
        job_items = list(TRANSLATION_JOBS.items())
    queued = running = done = failed = 0
    for _, future in job_items:
        if future.running():
            running += 1
        elif not future.done():
            queued += 1
        elif future.exception():
            failed += 1
        else:
            done += 1
    with get_db() as db:
        total = db.execute("SELECT COUNT(*) as c FROM papers").fetchone()["c"]
        translated = db.execute("""
            SELECT COUNT(*) as c FROM analyses
            WHERE translation IS NOT NULL AND translation != ''
        """).fetchone()["c"]
    return {
        "total_papers": total,
        "translated": translated,
        "missing": max(total - translated, 0),
        "jobs_total": len(job_items),
        "queued": queued,
        "running": running,
        "done": done,
        "failed": failed,
        "auto_translate_on_start": AUTO_TRANSLATE_ON_START,
        "auto_translate_limit": AUTO_TRANSLATE_LIMIT,
        "model": DEFAULT_LLM_MODEL,
        "llm_configured": bool(DEFAULT_LLM_API_KEY),
    }


def start_auto_translation_worker() -> None:
    global AUTO_TRANSLATE_STARTED
    if AUTO_TRANSLATE_STARTED or not AUTO_TRANSLATE_ON_START:
        return
    AUTO_TRANSLATE_STARTED = True

    def run() -> None:
        try:
            summary = queue_missing_translations(AUTO_TRANSLATE_LIMIT)
            log.info(f"Auto translation queued on startup: {summary}")
        except Exception as e:
            log.error(f"Auto translation startup queue failed: {e}")

    threading.Thread(target=run, name="auto_translation_startup", daemon=True).start()

# FastAPI
# ─────────────────────────────────────────────

app = FastAPI(title="Paper Reader API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NoteCreate(BaseModel):
    content: str

class NoteUpdate(BaseModel):
    content: str

class ReadingStatusUpdate(BaseModel):
    status: str  # unread / reading / read
    progress: float = 0.0


# ── 会议 & 同步 ──

SUPPORTED_CONFERENCES = [
    "WACV2026", "CVPR2025", "ICCV2025",
    "WACV2025", "CVPR2024", "ECCV2024", "ICCV2023",
    "WACV2024", "CVPR2023",
]

@app.get("/api/conferences")
def list_conferences():
    """列出支持的会议。"""
    with get_db() as db:
        synced = {r["conference"]: {"last_sync": r["last_sync"], "count": r["paper_count"]}
                  for r in db.execute("SELECT * FROM sync_log").fetchall()}
    return [{
        "id": c, "synced": c in synced,
        "paper_count": synced.get(c, {}).get("count", 0),
        "last_sync": synced.get(c, {}).get("last_sync"),
    } for c in SUPPORTED_CONFERENCES]


@app.post("/api/conferences/{conference}/sync")
def sync_conf(conference: str, force: bool = False):
    """同步会议论文。"""
    if conference not in SUPPORTED_CONFERENCES:
        raise HTTPException(404, f"不支持的会议: {conference}")
    count = sync_conference(conference, force=force)
    return {"conference": conference, "paper_count": count}


# ── 论文列表 & 搜索 ──

@app.get("/api/papers")
def list_papers(
    conference: Optional[str] = None,
    keyword: Optional[str] = None,
    status: Optional[str] = None,  # unread / reading / read
    offset: int = 0,
    limit: int = 20,
):
    """论文列表，支持会议过滤、关键词搜索、阅读状态过滤。"""
    with get_db() as db:
        conditions, params = [], []

        if conference:
            conditions.append("p.conference = ?")
            params.append(conference)
        if keyword:
            conditions.append("(p.title LIKE ? OR p.abstract LIKE ?)")
            kw = f"%{keyword}%"
            params.extend([kw, kw])
        if status:
            if status == "unread":
                conditions.append("(rs.status IS NULL OR rs.status = 'unread')")
            else:
                conditions.append("rs.status = ?")
                params.append(status)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        count_sql = f"""
            SELECT COUNT(*) as total FROM papers p
            LEFT JOIN reading_status rs ON p.id = rs.paper_id
            {where}
        """
        total = db.execute(count_sql, params).fetchone()["total"]

        sql = f"""
            SELECT p.*, rs.status as read_status, rs.progress,
                   (SELECT COUNT(*) FROM notes n WHERE n.paper_id = p.id) as note_count,
                   (CASE WHEN a.analysis IS NOT NULL AND a.analysis != '' THEN 1 ELSE 0 END) as has_analysis,
                   (CASE WHEN a.translation IS NOT NULL AND a.translation != '' THEN 1 ELSE 0 END) as has_translation
            FROM papers p
            LEFT JOIN reading_status rs ON p.id = rs.paper_id
            LEFT JOIN analyses a ON p.id = a.paper_id
            {where}
            ORDER BY p.created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = db.execute(sql, params).fetchall()

        papers = []
        for r in rows:
            papers.append({
                "id": r["id"],
                "conference": r["conference"],
                "title": r["title"],
                "authors": json.loads(r["authors"]) if r["authors"] else [],
                "abstract": r["abstract"] or "",
                "pdf_url": r["pdf_url"],
                "page_url": r["page_url"],
                "arxiv_url": r["arxiv_url"] or "",
                "pages": r["pages"] or "",
                "read_status": r["read_status"] or "unread",
                "progress": r["progress"] or 0.0,
                "note_count": r["note_count"],
                "has_analysis": bool(r["has_analysis"]),
                "has_translation": bool(r["has_translation"]),
            })

    return {"total": total, "offset": offset, "limit": limit, "papers": papers}


# ── 单篇论文 ──

@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str):
    """获取单篇论文详情（含 abstract、解读、笔记）。"""
    with get_db() as db:
        row = db.execute("""
            SELECT p.*, rs.status as read_status, rs.progress,
                   a.analysis, a.sections_json, a.model, a.token_count,
                   a.translation, a.translation_model, a.translation_token_count, a.translation_created_at
            FROM papers p
            LEFT JOIN reading_status rs ON p.id = rs.paper_id
            LEFT JOIN analyses a ON p.id = a.paper_id
            WHERE p.id = ?
        """, (paper_id,)).fetchone()

        if not row:
            raise HTTPException(404, "论文不存在")

        notes = db.execute(
            "SELECT * FROM notes WHERE paper_id=? ORDER BY created_at DESC", (paper_id,)
        ).fetchall()

    # 如果没有 abstract，尝试获取
    abstract = row["abstract"] or ""
    if not abstract:
        abstract = fetch_paper_abstract(paper_id) or ""

    sections = {}
    if row["sections_json"]:
        try:
            sections = json.loads(row["sections_json"])
        except:
            pass

    translation = row["translation"] or ""
    if translation and f"/api/papers/{paper_id}/figures/" not in translation:
        enriched_translation = insert_figure_markdown_by_refs(translation, paper_id)
        if enriched_translation != translation:
            translation = enriched_translation
            with get_db() as db:
                db.execute("UPDATE analyses SET translation=? WHERE paper_id=?", (translation, paper_id))

    return {
        "id": row["id"],
        "conference": row["conference"],
        "title": row["title"],
        "authors": json.loads(row["authors"]) if row["authors"] else [],
        "abstract": abstract,
        "pdf_url": row["pdf_url"],
        "page_url": row["page_url"],
        "arxiv_url": row["arxiv_url"] or "",
        "pages": row["pages"] or "",
        "read_status": row["read_status"] or "unread",
        "progress": row["progress"] or 0.0,
        "analysis": row["analysis"] or "",
        "translation": translation,
        "sections": sections,
        "llm_model": row["model"] or "",
        "token_count": row["token_count"] or 0,
        "translation_model": row["translation_model"] or "",
        "translation_token_count": row["translation_token_count"] or 0,
        "translation_created_at": row["translation_created_at"],
        "translation_job_status": _translation_job_status(paper_id),
        "notes": [{"id": n["id"], "content": n["content"],
                    "created_at": n["created_at"], "updated_at": n["updated_at"]}
                   for n in notes],
    }


# ── 解读 ──

@app.post("/api/papers/{paper_id}/analyze")
def trigger_analysis(paper_id: str):
    """触发 LLM 解读（异步友好：先提取 section，再调 LLM）。"""
    with get_db() as db:
        if not db.execute("SELECT 1 FROM papers WHERE id=?", (paper_id,)).fetchone():
            raise HTTPException(404, "论文不存在")

    sections = download_and_extract(paper_id)
    analysis = analyze_paper(paper_id)
    return {"paper_id": paper_id, "sections": list(sections.keys()),
            "analysis_length": len(analysis), "analysis": analysis}


@app.post("/api/papers/{paper_id}/translate")
def trigger_translation(paper_id: str, force: bool = False, background: bool = True):
    """Trigger Chinese LLM translation."""
    if background:
        return queue_translation(paper_id, force=force)

    with get_db() as db:
        if not db.execute("SELECT 1 FROM papers WHERE id=?", (paper_id,)).fetchone():
            raise HTTPException(404, "paper not found")
        if not force:
            row = db.execute("SELECT translation FROM analyses WHERE paper_id=?", (paper_id,)).fetchone()
            if row and row["translation"]:
                return {"paper_id": paper_id, "translation_length": len(row["translation"]), "translation": row["translation"], "cached": True}
        else:
            db.execute("""
                UPDATE analyses
                SET translation='', translation_model='', translation_token_count=0, translation_created_at=NULL
                WHERE paper_id=?
            """, (paper_id,))

    sections = download_and_extract(paper_id)
    translation = translate_paper(paper_id)
    return {"paper_id": paper_id, "sections": list(sections.keys()),
            "translation_length": len(translation), "translation": translation}


@app.get("/api/papers/{paper_id}/translate/status")
def get_translation_status(paper_id: str):
    """Get cached/background translation state."""
    with get_db() as db:
        row = db.execute("""
            SELECT translation, translation_model, translation_token_count, translation_created_at
            FROM analyses WHERE paper_id=?
        """, (paper_id,)).fetchone()
    if row and row["translation"]:
        return {
            "paper_id": paper_id,
            "status": "cached",
            "cached": True,
            "translation_length": len(row["translation"]),
            "translation_model": row["translation_model"] or "",
            "translation_token_count": row["translation_token_count"] or 0,
            "translation_created_at": row["translation_created_at"],
        }
    status = _translation_job_status(paper_id) or "missing"
    return {"paper_id": paper_id, "status": status, "cached": False}


@app.get("/api/papers/{paper_id}/translate/chunks")
def get_translation_chunks(paper_id: str):
    """Get per-chunk translation progress for retry/debug UI."""
    with get_db() as db:
        rows = db.execute("""
            SELECT chunk_index, chunk_total, source_label, status, error,
                   token_count, updated_at, length(translation) as translation_length
            FROM translation_chunks
            WHERE paper_id=?
            ORDER BY chunk_index
        """, (paper_id,)).fetchall()
    return {
        "paper_id": paper_id,
        "chunks": [dict(row) for row in rows],
    }


@app.get("/api/translations/status")
def get_translation_queue_status():
    """Get global translation queue/cache progress."""
    return translation_queue_summary()


@app.post("/api/translations/prefetch")
def prefetch_translations(limit: int = 0):
    """Queue missing translations without waiting for LLM completion."""
    summary = queue_missing_translations(limit)
    summary.update(translation_queue_summary())
    return summary


@app.get("/api/papers/{paper_id}/sections")
def get_sections(paper_id: str):
    """获取提取的 sections（不触发 LLM）。"""
    sections = download_and_extract(paper_id)
    return {"paper_id": paper_id, "sections": sections}


@app.get("/api/papers/{paper_id}/page-images")
def get_page_images(paper_id: str, limit: int = 8):
    """Render and list leading PDF page images for reading."""
    limit = max(1, min(limit, 20))
    paths = render_pdf_page_images(paper_id, max_pages=limit)
    return {
        "paper_id": paper_id,
        "images": [
            {
                "page": i + 1,
                "url": f"/api/papers/{paper_id}/page-images/{i + 1}",
            }
            for i, _ in enumerate(paths)
        ],
    }


@app.get("/api/papers/{paper_id}/page-images/{page_no}")
def get_page_image_file(paper_id: str, page_no: int):
    if page_no < 1:
        raise HTTPException(404, "page image not found")
    paths = render_pdf_page_images(paper_id, max_pages=page_no)
    if page_no > len(paths):
        raise HTTPException(404, "page image not found")
    return FileResponse(paths[page_no - 1], media_type="image/jpeg")


@app.get("/api/papers/{paper_id}/figures")
def get_figures(paper_id: str, limit: int = MAX_EXTRACTED_FIGURES):
    """Extract original figure crops near Fig./Figure captions."""
    limit = max(1, min(limit, 40))
    figures = extract_pdf_figures(paper_id, max_figures=limit)
    with get_db() as db:
        refs = {
            row["figure_no"]: dict(row)
            for row in db.execute("SELECT * FROM figure_refs WHERE paper_id=?", (paper_id,)).fetchall()
        }
    enriched = []
    for fig in figures:
        item = dict(fig)
        ref = refs.get(str(fig.get("number")))
        if ref:
            item.update({
                "first_ref_offset": ref.get("first_ref_offset", -1),
                "first_ref_text": ref.get("first_ref_text", ""),
                "confidence": ref.get("confidence", 0.0),
            })
        enriched.append(item)
    return {"paper_id": paper_id, "figures": enriched}


@app.get("/api/papers/{paper_id}/figures/{figure_no}")
def get_figure_file(paper_id: str, figure_no: str):
    figures = extract_pdf_figures(paper_id)
    for fig in figures:
        if str(fig.get("number")) == str(figure_no):
            path = FIGURE_DIR / paper_id / f"figure_{figure_no}.jpg"
            if path.exists():
                return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(404, "figure not found")


# ── 阅读状态 ──

@app.put("/api/papers/{paper_id}/status")
def update_reading_status(paper_id: str, body: ReadingStatusUpdate):
    with get_db() as db:
        db.execute("""
            INSERT OR REPLACE INTO reading_status (paper_id, status, progress, updated_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (paper_id, body.status, body.progress))
    return {"ok": True}


# ── 笔记 ──

@app.post("/api/papers/{paper_id}/notes")
def create_note(paper_id: str, body: NoteCreate):
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO notes (paper_id, content) VALUES (?, ?)",
            (paper_id, body.content))
        return {"id": cur.lastrowid, "paper_id": paper_id, "content": body.content}


@app.put("/api/notes/{note_id}")
def update_note(note_id: int, body: NoteUpdate):
    with get_db() as db:
        db.execute("UPDATE notes SET content=?, updated_at=datetime('now') WHERE id=?",
                   (body.content, note_id))
    return {"ok": True}


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int):
    with get_db() as db:
        db.execute("DELETE FROM notes WHERE id=?", (note_id,))
    return {"ok": True}


# ── 统计 ──

@app.get("/api/stats")
def get_stats():
    with get_db() as db:
        total = db.execute("SELECT COUNT(*) as c FROM papers").fetchone()["c"]
        read = db.execute("SELECT COUNT(*) as c FROM reading_status WHERE status='read'").fetchone()["c"]
        reading = db.execute("SELECT COUNT(*) as c FROM reading_status WHERE status='reading'").fetchone()["c"]
        analyzed = db.execute("SELECT COUNT(*) as c FROM analyses WHERE analysis != ''").fetchone()["c"]
        notes_count = db.execute("SELECT COUNT(*) as c FROM notes").fetchone()["c"]

        by_conf = db.execute("""
            SELECT conference, COUNT(*) as c FROM papers GROUP BY conference ORDER BY c DESC
        """).fetchall()

    return {
        "total_papers": total,
        "read": read,
        "reading": reading,
        "unread": total - read - reading,
        "analyzed": analyzed,
        "notes": notes_count,
        "by_conference": {r["conference"]: r["c"] for r in by_conf},
    }


# ── 启动 ──

@app.on_event("startup")
def startup():
    init_db()
    log.info(f"Paper Reader API 已启动 | DB: {DB_PATH}")
    start_auto_translation_worker()


if __name__ == "__main__":
    import uvicorn
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
