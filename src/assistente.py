"""
assistente.py - interface comum da v1 (TF-IDF) e da v2 (embeddings).

    resultado = assistente.responder("quanto tempo dura o cooldown?")
    resultado.resposta  -> str (ou mensagem de rejeição)
    resultado.id        -> int (-1 se rejeitado)
    resultado.escore    -> float
    resultado.ranking   -> list[tuple[int, float]]  (do mais ao menos similar)

Cada versão só precisa implementar _escores(texto), que devolve a
similaridade da pergunta com cada item da base. O resto (ranking por
resposta, empate, limiar, entradas estranhas) fica aqui, igual para as duas,
para que a comparação v1 x v2 seja justa.
"""

from dataclasses import dataclass, field

import numpy as np

MSG_REJEICAO = "Desculpe, não entendi. Pode reformular?"
MAX_CARACTERES = 1000   # textos gigantes são cortados (o TF-IDF e o modelo não precisam de mais)


@dataclass
class Resultado:
    resposta: str
    id: int
    escore: float
    ranking: list = field(default_factory=list)
    limiar: float = 0.0

    @property
    def rejeitado(self):
        return self.id == -1


def limpar_entrada(texto):
    """Qualquer coisa vira string; sem letra/número nenhum vira ''."""
    if texto is None:
        return ""
    texto = str(texto).strip()[:MAX_CARACTERES]
    if not any(c.isalnum() for c in texto):
        return ""
    return texto


def ranking_por_resposta(escores, ids, grupo_de):
    """Agrupa os itens que têm a mesma resposta (id canônico) ficando com o
    MAIOR escore do grupo, e ordena do mais ao menos similar.
    Empate de escore: ganha o menor id (regra determinística e documentada)."""
    melhor = {}
    for id_, s in zip(ids, escores):
        g = grupo_de[id_]
        s = float(s)
        if g not in melhor or s > melhor[g]:
            melhor[g] = s
    return sorted(melhor.items(), key=lambda par: (-round(par[1], 12), par[0]))


class AssistenteBase:
    nome = "base"

    def __init__(self, base, limiar):
        self.base = base
        self.limiar = float(limiar)

    def _escores(self, texto):
        raise NotImplementedError

    def responder(self, texto):
        texto = limpar_entrada(texto)
        if not texto:
            return Resultado(MSG_REJEICAO, -1, 0.0, [], self.limiar)

        escores = np.asarray(self._escores(texto), dtype=float)
        escores = np.nan_to_num(escores, nan=0.0)
        ranking = ranking_por_resposta(escores, self.base.ids, self.base.grupo_de)
        melhor_id, melhor_escore = ranking[0]

        if melhor_escore < self.limiar:
            return Resultado(MSG_REJEICAO, -1, melhor_escore, ranking, self.limiar)
        return Resultado(self.base.resposta(melhor_id), melhor_id, melhor_escore,
                         ranking, self.limiar)
