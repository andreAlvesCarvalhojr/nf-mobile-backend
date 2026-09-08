from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import HealthResponse, NfExtractionResult
from app.services.gemini_nf import GeminiNfService

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/pdf",
}

app = FastAPI(
    title="Despesas Backend",
    description="API local para extrair dados de nota fiscal com Gemini",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        gemini_configured=bool(settings.gemini_api_key),
        model=settings.gemini_model,
    )


@app.post("/extract-nf", response_model=NfExtractionResult)
async def extract_nf(file: UploadFile = File(...)) -> NfExtractionResult:
    mime_type = (file.content_type or "").lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {mime_type or 'desconhecido'}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Arquivo vazio")

    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Arquivo maior que 8 MB")

    try:
        service = GeminiNfService()
        return service.extract_from_image(image_bytes, mime_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Falha ao extrair dados da nota: {exc}",
        ) from exc
