from pydantic import BaseModel, Field


class NfExtractionResult(BaseModel):
    valor_total: float | None = Field(
        default=None,
        description="Valor total da nota em reais",
    )
    data_emissao: str | None = Field(
        default=None,
        description="Data de emissão no formato dd/MM/yyyy",
    )
    estabelecimento: str | None = Field(
        default=None,
        description="Nome do estabelecimento / emitente",
    )
    descricao_sugerida: str | None = Field(
        default=None,
        description="Sugestão curta de descrição para o lançamento",
    )
    chave_acesso: str | None = Field(
        default=None,
        description="Chave de acesso da NF-e/NFC-e com 44 dígitos, se legível",
    )
    cnpj: str | None = Field(
        default=None,
        description="CNPJ do emitente, se legível",
    )
    confianca: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Confiança geral da extração entre 0 e 1",
    )
    observacoes: str | None = Field(
        default=None,
        description="Observações ou campos duvidosos",
    )


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
    model: str
