import os
import io
import json
from typing import Any, Dict, List, Optional

# LlamaIndex for reading from persisted storage
from llama_index.core import StorageContext, load_index_from_storage

# ReportLab for quick PDF rendering
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

PROJECTS_DIR = "brd_projects"


class DocumentPreviewService:
    """
    Creates document previews from persisted LlamaIndex storage (embeddings).
    Does not rely on original files being present.

    Public:
        get_preview_text(project_id, document_id, max_chars=5000) -> str
        build_preview_pdf_bytes(text, title=None) -> bytes
    """

    # ---------- Public API ----------

    def get_preview_text(
        self,
        project_id: str,
        document_id: str,
        max_chars: int = 5000,
    ) -> str:
        """
        Collects representative text from the doc's persisted storage.
        Returns the first max_chars characters (cleaned).
        """
        metadata = self._load_project_metadata(project_id)
        doc = self._resolve_document(metadata, document_id)
        embedding_path = doc.get("embedding_path")
        if not embedding_path or not os.path.exists(embedding_path):
            return ""

        # 1) Try to read text directly out of docstore JSON files
        texts = self._extract_texts_from_docstore(embedding_path)

        # 2) Fallback: load the index and retrieve nodes with a generic query
        if not texts:
            texts = self._extract_texts_via_retriever(embedding_path)

        preview_text = self._sanitize(" \n".join(texts))[:max_chars].strip()
        return preview_text

    def build_preview_pdf_bytes(self, text: str, title: Optional[str] = None) -> bytes:
        """
        Render a simple, readable PDF from the preview text.
        """
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # Margins
        left = 0.8 * inch
        right = width - 0.8 * inch
        top = height - 0.8 * inch
        bottom = 0.8 * inch

        y = top

        # Title (optional)
        if title:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(left, y, f"Preview: {title}")
            y -= 0.35 * inch

        # Body
        c.setFont("Helvetica", 10)

        # Simple line wrap
        def wrap_line(s: str, max_chars: int = 100) -> List[str]:
            out: List[str] = []
            for para in s.splitlines():
                p = para.strip()
                if not p:
                    out.append("")
                    continue
                while len(p) > max_chars:
                    cut = p.rfind(" ", 0, max_chars)
                    cut = cut if cut != -1 else max_chars
                    out.append(p[:cut])
                    p = p[cut:].lstrip()
                out.append(p)
            return out

        for line in wrap_line(text or "No preview available.", 110):
            if y < bottom:
                c.showPage()
                y = top
                c.setFont("Helvetica", 10)
            c.drawString(left, y, line)
            y -= 14  # line height

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    # ---------- Helpers ----------

    def _load_project_metadata(self, project_id: str) -> Dict[str, Any]:
        path = os.path.join(PROJECTS_DIR, project_id, "metadata.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Project '{project_id}' not found")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_document(self, metadata: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        for d in metadata.get("documents", []):
            if d.get("id") == document_id:
                return d
        raise ValueError(f"Document '{document_id}' not found in project metadata")

    def _extract_texts_from_docstore(self, persist_dir: str) -> List[str]:
        """
        Attempts to pull text fields from JSON files within the persist dir.
        Defensive across different LlamaIndex versions.
        """
        candidates = []
        for fname in os.listdir(persist_dir):
            lower = fname.lower()
            if lower.endswith(".json") and any(k in lower for k in ["docstore", "store", "index"]):
                candidates.append(os.path.join(persist_dir, fname))

        texts: List[str] = []
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                self._collect_text_fields(data, texts, cap=2000)
                if texts:
                    break
            except Exception:
                continue
        return texts

    def _collect_text_fields(self, obj: Any, out: List[str], cap: int) -> None:
        if len(out) >= cap:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "text" and isinstance(v, str):
                    out.append(v)
                    if len(out) >= cap:
                        return
                else:
                    self._collect_text_fields(v, out, cap)
        elif isinstance(obj, list):
            for v in obj:
                self._collect_text_fields(v, out, cap)

    def _extract_texts_via_retriever(self, persist_dir: str) -> List[str]:
        texts: List[str] = []
        try:
            storage = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage)
            retriever = index.as_retriever(similarity_top_k=25)
            nodes = retriever.retrieve("document preview")
            for n in nodes:
                # Compatible with different Node shapes
                t = getattr(n, "text", None)
                if not isinstance(t, str):
                    # some versions expose .get_text()
                    t = getattr(n, "get_text", lambda: "")()
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        except Exception:
            pass
        return texts

    def _sanitize(self, s: str) -> str:
        return s.replace("\x00", "").replace("\u0000", "")