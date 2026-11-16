"""
base.py - base de conhecimento (dados/base.csv) e conjuntos de validação/teste.

Decisões tratadas aqui:
  * Respostas duplicadas: perguntas diferentes da base podem ter exatamente a
    mesma resposta (ex.: ids 11 e 12). Todas apontam para um "id canônico"
    (o menor id do grupo). O ranking dos assistentes é feito por resposta,
    não por pergunta, para que a mesma resposta não ocupe duas posições
    (isso distorceria o MRR).
  * Dicionário de siglas: tokens em CAIXA ALTA encontrados na base (VAC, VRS,
    MR12, CT...) viram um conjunto usado pela normalização para proteger
    siglas mesmo quando o usuário digita em minúsculas.
"""

from pathlib import Path

import pandas as pd

from lexer_afd import TipoToken, tokenizar

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
CAMPOS = ("pergunta", "resposta", "pergunta+resposta")


class Base:
    def __init__(self, df):
        obrigatorias = {"id", "pergunta", "resposta", "fonte"}
        faltando = obrigatorias - set(df.columns)
        if faltando:
            raise ValueError(f"base.csv sem as colunas {faltando}")
        if df["id"].duplicated().any():
            raise ValueError("base.csv tem ids repetidos")
        self.df = df.sort_values("id").reset_index(drop=True)
        self.ids = self.df["id"].astype(int).tolist()
        self._por_id = self.df.set_index("id")

        # id canônico de cada resposta (menor id que tem aquele texto)
        canonico_da_resposta = {}
        self.grupo_de = {}
        for i, resp in zip(self.ids, self.df["resposta"]):
            chave = " ".join(str(resp).split()).lower()
            canonico_da_resposta.setdefault(chave, i)
            self.grupo_de[i] = canonico_da_resposta[chave]

        self.siglas = self._extrair_siglas()

    def __len__(self):
        return len(self.ids)

    def _extrair_siglas(self):
        siglas = set()
        for texto in pd.concat([self.df["pergunta"], self.df["resposta"]]):
            for tok in tokenizar(str(texto)):
                if tok.tipo == TipoToken.SIGLA:
                    lex = tok.lexema
                    if lex.endswith("s") and len(lex) > 2:
                        lex = lex[:-1]
                    siglas.add(lex.lower())
        return siglas

    def textos(self, campo="pergunta"):
        if campo == "pergunta":
            return self.df["pergunta"].tolist()
        if campo == "resposta":
            return self.df["resposta"].tolist()
        if campo == "pergunta+resposta":
            return (self.df["pergunta"] + " " + self.df["resposta"]).tolist()
        raise ValueError(f"campo deve ser um de {CAMPOS}")

    def pergunta(self, id_):
        return self._por_id.loc[id_, "pergunta"]

    def resposta(self, id_):
        return self._por_id.loc[id_, "resposta"]

    def fonte(self, id_):
        return self._por_id.loc[id_, "fonte"]

    def canonico(self, id_):
        """-1 continua -1 (fora do escopo)."""
        return self.grupo_de.get(int(id_), -1) if int(id_) != -1 else -1


def carregar_base(caminho=None):
    caminho = Path(caminho) if caminho else DADOS / "base.csv"
    return Base(pd.read_csv(caminho))


def carregar_perguntas(nome):
    """nome = 'validacao' ou 'teste' (ou um caminho para csv com pergunta,id_esperado)."""
    caminho = Path(nome) if str(nome).endswith(".csv") else DADOS / f"{nome}.csv"
    df = pd.read_csv(caminho)
    df["id_esperado"] = df["id_esperado"].astype(int)
    return df


def _simplificar(texto):
    texto = "".join(c for c in str(texto).lower() if c.isalnum() or c.isspace())
    return " ".join(texto.split())


def checar_contaminacao(base, perguntas):
    """Lista perguntas que aparecem literalmente na base (ignorando caixa,
    pontuação e espaços). Teste contaminado anula o critério 5."""
    da_base = {_simplificar(p) for p in base.textos("pergunta")}
    return [p for p in perguntas["pergunta"] if _simplificar(p) in da_base]
