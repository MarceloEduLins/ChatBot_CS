"""
chatbot_denso.py - v2: recuperação com sentence embeddings pré-treinados.

Mesma arquitetura da v1 (ranking por cosseno, limiar, mesma interface); só
muda a representação: em vez de um vetor esparso de contagens (TF-IDF), cada
frase vira um vetor denso de 384 dimensões produzido por um Transformer.

Modelo padrão: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - leve (~118M parâmetros), roda em CPU, treinado para paráfrases em 50+
    línguas, inclusive português.
Alternativa comparada: paraphrase-multilingual-mpnet-base-v2 (maior, 768 dim).

Os embeddings da base são calculados uma vez e salvos em cache/ (.npy). A
chave do arquivo depende do modelo, dos textos e de normalizar ou não, então
se a base mudar o cache é refeito sozinho.

Abreviações de chat ("vc", "qnto", "n") são expandidas na pergunta do usuário
antes do embedding (expandir_abreviacoes=True). A base não tem abreviações, então
o texto da base e o cache não mudam.
"""

import hashlib
import json
from pathlib import Path

from assistente import AssistenteBase
from base import RAIZ
from normalizacao import ConfigNormalizacao, Normalizador, expandir_abreviacoes
from similaridade import cosseno

MODELO_PADRAO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_ALTERNATIVO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LIMIAR_PADRAO_V2 = 0.60   # só vale até rodar a avaliação
PASTA_CACHE = RAIZ / "cache"

_MODELOS = {}   # evita carregar o mesmo modelo duas vezes na avaliação


def carregar_modelo(nome):
    if nome not in _MODELOS:
        from sentence_transformers import SentenceTransformer
        _MODELOS[nome] = SentenceTransformer(nome, device="cpu")
    return _MODELOS[nome]


def info_modelo(nome):
    """Versão do modelo para registrar nos resultados (reprodutibilidade)."""
    info = {"modelo": nome}
    try:
        import sentence_transformers
        import torch
        info["sentence_transformers"] = sentence_transformers.__version__
        info["torch"] = torch.__version__
        modelo = carregar_modelo(nome)
        info["dimensao"] = modelo.get_sentence_embedding_dimension()
        commit = getattr(modelo[0].auto_model.config, "_commit_hash", None)
        if not commit:   # versões novas do transformers não preenchem _commit_hash
            from huggingface_hub import constants
            ref = Path(constants.HF_HUB_CACHE) / f"models--{nome.replace('/', '--')}" / "refs" / "main"
            commit = ref.read_text(encoding="utf-8").strip() if ref.exists() else None
        if commit:
            info["revisao_hf"] = commit
    except Exception as erro:  # não pode derrubar a avaliação
        info["erro_info"] = str(erro)
    return info


class AssistenteDenso(AssistenteBase):
    nome = "v2"

    def __init__(self, base, limiar=LIMIAR_PADRAO_V2, modelo=MODELO_PADRAO,
                 normalizar=False, campo="pergunta", config=None, pasta_cache=PASTA_CACHE,
                 expandir_abreviacoes=True):
        super().__init__(base, limiar)
        self.nome_modelo = modelo
        self.normalizar = normalizar
        self.expandir_abreviacoes = expandir_abreviacoes
        self.campo = campo
        self.modelo = carregar_modelo(modelo)
        self.normalizador = Normalizador(config or ConfigNormalizacao(), siglas=base.siglas)
        self.pasta_cache = Path(pasta_cache)
        textos = [self._preparar(t) for t in base.textos(campo)]
        self.matriz = self._embeddings_da_base(textos)

    def _preparar(self, texto):
        """Com normalizar=True o modelo recebe os tokens do v0 juntos por espaço."""
        if self.normalizar:
            return " ".join(self.normalizador.normalizar(texto))
        return texto

    def _embeddings_da_base(self, textos):
        import numpy as np
        conteudo = json.dumps([self.nome_modelo, self.normalizar, textos], ensure_ascii=False)
        chave = hashlib.sha1(conteudo.encode("utf-8")).hexdigest()[:12]
        apelido = self.nome_modelo.split("/")[-1]
        arquivo = self.pasta_cache / f"emb_{apelido}_{'norm' if self.normalizar else 'cru'}_{chave}.npy"
        if arquivo.exists():
            return np.load(arquivo)
        self.pasta_cache.mkdir(parents=True, exist_ok=True)
        emb = self.modelo.encode(textos, batch_size=32, convert_to_numpy=True,
                                 show_progress_bar=False)
        np.save(arquivo, emb)
        return emb

    def _escores(self, texto):
        import numpy as np
        if self.expandir_abreviacoes and not self.normalizar:
            texto = expandir_abreviacoes(texto)   # com normalizar=True o v0 já expande
        texto = self._preparar(texto)
        if not texto.strip():           # ex.: só stopwords com normalizar=True
            return np.zeros(len(self.base))
        q = self.modelo.encode([texto], convert_to_numpy=True, show_progress_bar=False)[0]
        return cosseno(q, self.matriz)


if __name__ == "__main__":
    from base import carregar_base
    assistente = AssistenteDenso(carregar_base())
    for p in ["quanto tempo dura o cooldown?", "qual a senha do wifi?"]:
        r = assistente.responder(p)
        print(p, "->", r.id, round(r.escore, 3), r.resposta[:80])
