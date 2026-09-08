from __future__ import annotations

import json
import re

from google import genai
from google.genai import types

from app.config import settings
from app.schemas import NfExtractionResult

EXTRACTION_PROMPT = """
Você é um extrator de dados de nota fiscal brasileira (NF-e / NFC-e / cupom fiscal).

Analise a imagem e retorne APENAS um JSON válido, sem markdown, com este formato:
{
  "valor_total": 123.45,
  "data_emissao": "dd/MM/yyyy",
  "estabelecimento": "nome do emitente",
  "descricao_sugerida": "descrição curta para despesa",
  "chave_acesso": "44 digitos ou null",
  "cnpj": "somente digitos ou null",
  "confianca": 0.0,
  "observacoes": "texto ou null"
}

Regras:
- valor_total deve ser número decimal com ponto
- data_emissao deve estar em dd/MM/yyyy
- chave_acesso só se tiver 44 dígitos legíveis
- se um campo não estiver legível, use null
- confianca entre 0 e 1
""".strip()


class GeminiNfService:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY não configurada. Copie .env.example para .env e preencha a chave."
            )
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def extract_from_image(self, image_bytes: bytes, mime_type: str) -> NfExtractionResult:
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part.from_text(text=EXTRACTION_PROMPT),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        raw_text = (response.text or "").strip()
        payload = self._parse_json(raw_text)
        return NfExtractionResult.model_validate(payload)

    @staticmethod
    def _parse_json(raw_text: str) -> dict:
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Resposta do Gemini não contém JSON válido: {raw_text[:300]}")

        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("JSON do Gemini não é um objeto")
        return data
