"""File ingestion: turn an uploaded file into searchable text chunks.

    PDF / Word / text / CSV  -> text is extracted locally (scanned PDF pages are read by a vision model)
    image (png, jpg, webp)   -> a vision model describes it and transcribes visible text
    audio (mp3, wav, m4a...) -> Whisper transcribes it

Only the extracted text is stored (as Chunk rows). The original file is not kept.
Vision and Whisper run on Groq and need GROQ_API_KEY on the server.
"""
import base64
import io
import logging
import os
import re
import zipfile
from html import unescape

from django.conf import settings
from django.db import transaction

from .models import Chunk, Upload

log = logging.getLogger(__name__)

MB = 1024 * 1024
LIMITS = {"text": 10 * MB, "image": 12 * MB, "audio": 24 * MB}   # bytes per file
EXTENSIONS = {
    "text": {".pdf", ".docx", ".txt", ".md", ".csv"},
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif"},
    "audio": {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mpga"},
}
MAX_PDF_PAGES = 150        # text pages read per PDF
MAX_OCR_PAGES = 8          # scanned pages sent to the vision model per PDF
MAX_CHUNKS_PER_FILE = 500
MAX_TOTAL_CHUNKS = 4000    # keeps the in-memory BM25 index small on a tiny server
CHUNK_CHARS = 700
OVERLAP_CHARS = 100

IMAGE_PROMPT = ("Describe this image for a search index. State what it shows, then transcribe ALL visible text "
                "exactly, and list any numbers, tables, labels or names. Plain text only, no markdown.")
OCR_PROMPT = "Transcribe all the text on this page exactly as written. Output only the text, no commentary."


class IngestError(Exception):
    """A problem the user can understand and fix (wrong type, too big, unreadable, missing key)."""


# ── helpers ──────────────────────────────────────────────────────────────────
def clean_name(name):
    name = os.path.basename(name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return (name or "file")[:150]


def kind_of(name):
    ext = os.path.splitext(name)[1].lower()
    for kind, exts in EXTENSIONS.items():
        if ext in exts:
            return kind
    return None


def supported_types():
    return {k: ", ".join(sorted(e.lstrip(".") for e in v)) for k, v in EXTENSIONS.items()}


def groq_ready():
    return bool(os.environ.get("GROQ_API_KEY"))


def chunk_text(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """Split text into ~size-character chunks on paragraph/sentence boundaries, with a small overlap."""
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        if not para:
            continue
        if len(para) <= size:
            pieces.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > size:       # a single huge "sentence": hard split
                pieces.append(sent[:size])
                sent = sent[size:]
            if sent:
                pieces.append(sent)
    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else ""
            cur = f"{tail} {piece}".strip()
        else:
            cur = f"{cur} {piece}".strip() if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def _strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return re.sub(r"<think>.*", "", text, flags=re.S).strip()


# ── Groq calls (vision + speech) ─────────────────────────────────────────────
def _client():
    if not groq_ready():
        raise IngestError("This server has no GROQ_API_KEY, so it cannot read images or audio with a model.")
    from groq import Groq
    return Groq(timeout=60)


def _api_error(what, exc):
    status = getattr(exc, "status_code", None)
    hint = {401: "The Groq key was rejected.", 404: "The model name was not found.", 429: "Groq rate limit reached; try again shortly.",
            413: "The file is too large for the model."}.get(status, "Check GROQ_API_KEY and the model settings (VISION_MODEL / WHISPER_MODEL).")
    return f"The {what} call failed{f' (HTTP {status})' if status else ''}. {hint}"


def _vision(data, mime, prompt):
    client = _client()
    b64 = base64.b64encode(data).decode()
    try:
        resp = client.chat.completions.create(
            model=settings.VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            temperature=0.1, max_completion_tokens=1500)
    except Exception as exc:
        log.exception("vision call failed")
        raise IngestError(_api_error("vision model", exc))
    return _strip_think(resp.choices[0].message.content)


def _transcribe(data, filename):
    client = _client()
    try:
        resp = client.audio.transcriptions.create(file=(filename, data), model=settings.WHISPER_MODEL)
    except Exception as exc:
        log.exception("transcription failed")
        raise IngestError(_api_error("speech-to-text", exc))
    return (getattr(resp, "text", None) or str(resp)).strip()


# ── extractors ───────────────────────────────────────────────────────────────
def _plain(data):
    if b"\x00" in data[:4096]:
        raise IngestError("This looks like a binary file, not text.")
    return data.decode("utf-8-sig", errors="replace")


def _docx(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > 40 * MB:
                raise IngestError("This Word file is too large to read safely.")
            xml = z.read(info).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError):
        raise IngestError("This is not a valid .docx file.")
    xml = re.sub(r"</w:p>", "\n\n", xml)
    xml = re.sub(r"<w:(tab|br)\s*/>", " ", xml)
    return unescape(re.sub(r"<[^>]+>", "", xml))


def _pdf(data):
    """Return ([(label, text)], note). Pages without a text layer go to the vision model (OCR) when it is available."""
    from pypdf import PdfReader
    if b"%PDF-" not in data[:1024]:
        raise IngestError("This file is not a real PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestError("This PDF is password protected.")
        total = len(reader.pages)
    except IngestError:
        raise
    except Exception:
        raise IngestError("Could not open this PDF (it may be damaged).")
    pages, scanned, ocr_done, ocr_failed = [], 0, 0, 0
    for i, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if len(text) >= 20:
            pages.append((f"p{i}", text))
            continue
        scanned += 1
        if not groq_ready() or ocr_done >= MAX_OCR_PAGES:
            continue
        try:
            img = page.images[0]
            ext = os.path.splitext(img.name)[1].lower()
            raw, mime = _prepare_image(img.data, "image/png" if ext == ".png" else "image/jpeg")
            ocr = _vision(raw, mime, OCR_PROMPT)
            ocr_done += 1
            if ocr:
                pages.append((f"p{i} (OCR)", ocr))
        except IngestError:
            ocr_failed += 1
        except Exception:
            ocr_failed += 1
    notes = [f"PDF, {total} page(s)"]
    if total > MAX_PDF_PAGES:
        notes.append(f"only the first {MAX_PDF_PAGES} pages were read")
    if ocr_done:
        notes.append(f"{ocr_done} scanned page(s) read with the vision model")
    skipped = scanned - ocr_done
    if skipped > 0:
        why = "set GROQ_API_KEY to read scanned pages" if not groq_ready() else f"OCR is limited to {MAX_OCR_PAGES} pages per file" if ocr_failed == 0 else "OCR failed for these"
        notes.append(f"{skipped} page(s) had no text and were skipped ({why})")
    return pages, "; ".join(notes)


MAX_IMAGE_PIXELS = 40_000_000   # about 8000 x 5000; bigger images can exhaust a small server's memory


def _flatten(im):
    """Return an 8-bit RGB copy: transparency is placed on white (black-on-transparent diagrams stay readable)."""
    from PIL import Image
    if im.mode in ("I", "I;16", "I;16L", "I;16B", "F"):
        im = im.convert("I").point(lambda v: v * (1 / 256)).convert("L")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def _prepare_image(data, mime="image/jpeg"):
    """Shrink big photos so the request stays small (Groq limits base64 image requests to 4 MB)."""
    try:
        from PIL import Image, ImageFile
    except ImportError:
        if len(data) > int(2.8 * MB):
            raise IngestError("Image is over 2.8 MB and image resizing is not installed.")
        return data, mime
    ImageFile.LOAD_TRUNCATED_IMAGES = True     # accept slightly damaged files
    Image.MAX_IMAGE_PIXELS = 400_000_000       # we do our own, stricter check below
    try:
        im = Image.open(io.BytesIO(data))
        width, height = im.size
        if width * height > MAX_IMAGE_PIXELS:
            raise IngestError(f"This image is very large ({width}x{height}). Shrink it to under 40 megapixels and upload again.")
        try:
            im.draft("RGB", (1600, 1600))      # JPEG: decode at reduced size
        except Exception:
            pass
        im = _flatten(im)
        im.thumbnail((1600, 1600))
        for quality in (85, 65, 45):
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
            if buf.tell() < int(2.8 * MB):
                return buf.getvalue(), "image/jpeg"
        raise IngestError("This image is too large even after shrinking.")
    except IngestError:
        raise
    except MemoryError:
        raise IngestError("The server ran out of memory reading this image. Shrink it and try again.")
    except Exception as exc:
        log.exception("could not read image")
        raise IngestError(f"Could not read this image ({type(exc).__name__}: {str(exc)[:100]}).")


def extract(name, kind, data, description=""):
    """Return ([(title, chunk_text)], note) for one file."""
    ext = os.path.splitext(name)[1].lower()
    if kind == "text":
        if ext == ".pdf":
            pages, note = _pdf(data)
        elif ext == ".docx":
            pages, note = [("", _docx(data))], "Word document"
        else:
            pages, note = [("", _plain(data))], "text file"
    elif kind == "image":
        pages, note = [("", _model_text(description, lambda: _vision(*_prepare_image(data), IMAGE_PROMPT), "image"))], \
            "image described by the vision model" if groq_ready() else "image: used your description"
    else:
        pages, note = [("", _model_text(description, lambda: _transcribe(data, name), "audio"))], \
            "audio transcribed by Whisper" if groq_ready() else "audio: used your transcript"
    parts = []
    for label, text in pages:
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            if label:
                title = f"{name} · {label}"
            elif len(chunks) > 1:
                title = f"{name} · part {i + 1}"
            else:
                title = name
            parts.append((title, chunk))
    if not parts:
        raise IngestError("No readable text found in this file.")
    if len(parts) > MAX_CHUNKS_PER_FILE:
        parts = parts[:MAX_CHUNKS_PER_FILE]
        note += f"; truncated to {MAX_CHUNKS_PER_FILE} chunks"
    return parts, note


def _model_text(description, call, what):
    """Run a vision/speech model; if it is unavailable or fails, fall back to the text the user typed."""
    description = (description or "").strip()
    try:
        text = call()
    except IngestError:
        if description:
            return description
        raise
    if not text:
        if description:
            return description
        raise IngestError(f"The model returned nothing for this {what}.")
    return f"{description}. {text}" if description else text


# ── storage ──────────────────────────────────────────────────────────────────
def save_upload(f, acl="public", description=""):
    """Validate, extract and store one uploaded file. Returns (Upload, chunk_count)."""
    name = clean_name(f.name)
    kind = kind_of(name)
    if not kind:
        allowed = ", ".join(sorted(e for v in EXTENSIONS.values() for e in v))
        raise IngestError(f"Unsupported file type. Allowed: {allowed}")
    if f.size > LIMITS[kind]:
        raise IngestError(f"File is too big (limit {LIMITS[kind] // MB} MB for {kind} files).")
    if Chunk.objects.count() >= MAX_TOTAL_CHUNKS:
        raise IngestError("The knowledge base is full. Delete some files first.")
    parts, note = extract(name, kind, f.read(), description)
    acl = acl if acl in ("public", "staff") else "public"
    with transaction.atomic():
        upload = Upload.objects.create(filename=name, kind=kind, acl=acl, note=note[:300])
        Chunk.objects.bulk_create([Chunk(upload=upload, idx=i, title=t[:300], text=c) for i, (t, c) in enumerate(parts)])
    return upload, len(parts)
