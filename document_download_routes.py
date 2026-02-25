# document_download_routes.py

from fastapi import APIRouter, HTTPException

from fastapi.responses import Response, PlainTextResponse

from services.document_preview_service import DocumentPreviewService

import re



doc_download_router = APIRouter(prefix="/projects/{project_id}/documents", tags=["document-download"])

svc = DocumentPreviewService()





def _safe_filename(name: str) -> str:

    # Strip unsafe chars

    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip("_") or "document"





@doc_download_router.get("/{document_id}/download")

def download_document_preview(project_id: str, document_id: str, format: str = "pdf", max_chars: int = 5000):

    """

    Returns an on-the-fly preview for a document.

    - format=pdf (default): inline PDF preview

    - format=txt        : plain text

    """

    try:

        text = svc.get_preview_text(project_id=project_id, document_id=document_id, max_chars=max_chars)

    except Exception as e:

        raise HTTPException(status_code=404, detail=str(e))



    # Reserve minimal content to avoid empty viewers

    if not text:

        text = "No preview available."



    # Determine a friendly filename (use the original if we can)

    try:

        # quick lookup for filename

        metadata = svc._load_project_metadata(project_id)  # read-only use

        filename = next((d.get("filename") for d in metadata.get("documents", []) if d.get("id") == document_id), None)

    except Exception:

        filename = None



    base = _safe_filename(filename or f"{document_id}")

    if format.lower() == "txt":

        disp_name = f"{base}.preview.txt"

        return PlainTextResponse(

            content=text,

            headers={"Content-Disposition": f'inline; filename="{disp_name}"'},

            media_type="text/plain; charset=utf-8",

        )



    # Default: PDF

    pdf_bytes = svc.build_preview_pdf_bytes(text=text, title=filename or document_id)

    disp_name = f"{base}.preview.pdf"

    return Response(

        content=pdf_bytes,

        media_type="application/pdf",

        headers={"Content-Disposition": f'inline; filename="{disp_name}"'},

    )
 