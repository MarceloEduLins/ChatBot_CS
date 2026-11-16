"""
normalizacao.py - v0: pipeline de normalização de textos em português.

Função principal: normalizar(texto, **etapas) -> list[str]

Etapas (cada uma liga/desliga por parâmetro, para o experimento de ablação):
  0. expansão de abreviações  "vc" -> "você", "qnto" -> "quanto", "n" -> "não"...
                              (dicionário ABREVIACOES; aplicada no texto, antes de tudo)
  1. tokenização ............ NLTK (word_tokenize) ou o nosso lexer AFD
  2. minúsculas
  3. remoção de pontuação ... e de símbolos fora do alfabeto (emoji etc.)
  4. remoção de stopwords ... lista do NLTK, com ajustes (negação e
                              interrogativos podem ser mantidos)
  5. stemming ............... RSLPStemmer do NLTK
  6. remoção de acentos ..... feita DEPOIS do stemming, porque as regras do
                              RSLP usam os sufixos acentuados (-ção, -ões...)
Extras do grupo:
  - preservar_siglas: siglas (VAC, VRS, MR12...) não passam por stopwords nem
    stemming. Uma sigla é reconhecida pela caixa (AFD do lexer) OU por estar
    no dicionário de siglas extraído da base (para pegar "vac" digitado em
    minúsculas).
  - marcar_negacao: a palavra de conteúdo logo depois de "não/nem/nunca/sem"
    ganha o prefixo "nao_" (ex.: "não consigo" -> ["não", "nao_consig"]).

Decisões e justificativas estão em docs/decisoes.md.
"""

import re
import unicodedata
from dataclasses import dataclass, asdict

from lexer_afd import TipoToken, classificar, tokenizar as tokenizar_afd

NEGACOES = {"não", "nao", "nem", "nunca", "jamais", "sem", "nenhum", "nenhuma"}
INTERROGATIVOS = {"que", "quanto", "quanta", "quantos", "quantas", "quando",
                  "como", "qual", "quais", "onde", "quem", "porque", "por que"}


# Abreviações comuns de chat. Só entram formas sem ambiguidade no nosso domínio:
# "to" (inglês / "estou") e "num" ("em um" / "não") ficaram de fora de propósito.
ABREVIACOES = {
    "vc": "você", "vcs": "vocês", "pq": "por que", "q": "que", "oq": "o que",
    "qnto": "quanto", "qnt": "quanto", "qnta": "quanta", "qntos": "quantos", "qntas": "quantas",
    "qnd": "quando", "qndo": "quando", "qdo": "quando", "ql": "qual", "qls": "quais",
    "tb": "também", "tbm": "também", "tmb": "também", "n": "não", "ñ": "não", "eh": "é",
    "d": "de", "pra": "para", "pro": "para o", "pros": "para os", "pras": "para as",
    "td": "tudo", "tds": "todos", "mt": "muito", "mto": "muito", "mts": "muitos",
    "msm": "mesmo", "cmg": "comigo", "hj": "hoje", "agr": "agora", "dps": "depois",
    "ta": "está", "tá": "está", "tava": "estava", "tô": "estou", "ngm": "ninguém",
    "nd": "nada", "vdd": "verdade", "msg": "mensagem", "kd": "cadê", "pfv": "por favor",
}
_PADRAO_ABREV = re.compile(
    r"(?<![\w-])(" + "|".join(sorted(map(re.escape, ABREVIACOES), key=len, reverse=True))
    + r")(?![\w-])", re.IGNORECASE)


def expandir_abreviacoes(texto):
    """Troca abreviações de chat pela palavra inteira, só quando são palavras
    soltas: "vc tb" -> "você também", mas "praia", "3d" e "pick-ban" não mudam."""
    return _PADRAO_ABREV.sub(lambda m: ABREVIACOES[m.group(1).lower()], texto)


@dataclass
class ConfigNormalizacao:
    tokenizador: str = "nltk"          # "nltk" ou "afd"
    minusculas: bool = True
    remover_pontuacao: bool = True
    remover_stopwords: bool = True
    manter_negacao: bool = True         # tira as negações da lista de stopwords
    manter_interrogativos: bool = False
    marcar_negacao: bool = False
    stemming: bool = True
    remover_acentos: bool = True
    preservar_siglas: bool = True
    expandir_abreviacoes: bool = True   # etapa 0 (dicionário ABREVIACOES)

    def como_dict(self):
        return asdict(self)


def remover_acentos(palavra):
    nfd = unicodedata.normalize("NFD", palavra)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------
# Recursos do NLTK (carregados uma vez só)
# ---------------------------------------------------------------
_RECURSOS = {}


def _garantir_nltk(recurso, caminho):
    import nltk
    try:
        nltk.data.find(caminho)
    except LookupError:
        nltk.download(recurso, quiet=True)
        nltk.data.find(caminho)  # se ainda faltar, deixa o erro aparecer


def _stopwords_nltk():
    if "stop" not in _RECURSOS:
        from nltk.corpus import stopwords
        _garantir_nltk("stopwords", "corpora/stopwords")
        _RECURSOS["stop"] = set(stopwords.words("portuguese"))
    return _RECURSOS["stop"]


def _stemmer():
    if "rslp" not in _RECURSOS:
        from nltk.stem import RSLPStemmer
        _garantir_nltk("rslp", "stemmers/rslp")
        _RECURSOS["rslp"] = RSLPStemmer()
    return _RECURSOS["rslp"]


def _tokenizar_nltk(texto):
    from nltk.tokenize import word_tokenize
    if "punkt" not in _RECURSOS:
        _garantir_nltk("punkt_tab", "tokenizers/punkt_tab")
        _RECURSOS["punkt"] = True
    return word_tokenize(texto, language="portuguese")


# ---------------------------------------------------------------
# Normalizador
# ---------------------------------------------------------------
class Normalizador:
    """Guarda a configuração e o dicionário de siglas da base. Um objeto
    Normalizador pode ser passado direto como `analyzer` do TfidfVectorizer."""

    def __init__(self, config=None, siglas=None):
        self.config = config or ConfigNormalizacao()
        self.siglas = {s.lower() for s in (siglas or set())}
        self.stoplist = self._montar_stoplist() if self.config.remover_stopwords else set()

    def _montar_stoplist(self):
        stop = set(_stopwords_nltk())
        if self.config.manter_negacao:
            stop -= NEGACOES
        if self.config.manter_interrogativos:
            stop -= INTERROGATIVOS
        # inclui as versões sem acento ("nao", "voce") para pegar quem digita sem acento
        stop |= {remover_acentos(p) for p in stop}
        if self.config.manter_negacao:
            stop -= NEGACOES
        if self.config.manter_interrogativos:
            stop -= INTERROGATIVOS
        return stop

    def _tokens(self, texto):
        """Devolve pares (lexema, tipo)."""
        if self.config.tokenizador == "afd":
            return [(t.lexema, t.tipo) for t in tokenizar_afd(texto)]
        return [(lex, classificar(lex)) for lex in _tokenizar_nltk(texto)]

    def __call__(self, texto):
        return self.normalizar(texto)

    def normalizar(self, texto):
        cfg = self.config
        if not isinstance(texto, str) or not texto.strip():
            return []
        if cfg.expandir_abreviacoes:
            texto = expandir_abreviacoes(texto)

        # Texto todo em CAPS LOCK: a caixa não indica sigla, então só vale o dicionário
        letras = [c for c in texto if c.isalpha()]
        caps_lock = len(letras) >= 12 and sum(c.isupper() for c in letras) / len(letras) > 0.8

        saida = []
        negar_proximo = False
        for lexema, tipo in self._tokens(texto):
            if tipo in (TipoToken.PONTUACAO, TipoToken.INVALIDO):
                negar_proximo = False   # o escopo da negação termina na pontuação
                if cfg.remover_pontuacao:
                    continue
                saida.append(lexema)
                continue

            minuscula = lexema.lower()
            # plural de sigla: CTs -> ct ; "cts" digitado em minúsculas -> ct
            if tipo == TipoToken.SIGLA and lexema.endswith("s") and len(lexema) > 2:
                minuscula = minuscula[:-1]
            elif (minuscula.endswith("s") and minuscula not in self.siglas
                  and len(minuscula) > 2 and minuscula[:-1] in self.siglas):
                minuscula = minuscula[:-1]
            sigla = cfg.preservar_siglas and (
                (tipo == TipoToken.SIGLA and not caps_lock) or minuscula in self.siglas)

            token = minuscula if (cfg.minusculas or sigla) else lexema

            # negação: decide antes de jogar a stopword fora
            eh_negacao = minuscula in NEGACOES
            if eh_negacao and cfg.marcar_negacao:
                negar_proximo = True

            if cfg.remover_stopwords and not sigla and minuscula in self.stoplist:
                continue

            if cfg.stemming and not sigla and tipo == TipoToken.PALAVRA:
                token = _stemmer().stem(token.lower())

            if cfg.remover_acentos:
                token = remover_acentos(token)

            if negar_proximo and not eh_negacao:
                token = "nao_" + token
                negar_proximo = False

            if token:
                saida.append(token)
        return saida


def normalizar(texto, siglas=None, **etapas):
    """Atalho pedido no enunciado: normalizar(texto) -> list[str].
    Ex.: normalizar("Não consigo jogar Premier", stemming=False)"""
    return Normalizador(ConfigNormalizacao(**etapas), siglas=siglas).normalizar(texto)


if __name__ == "__main__":
    import sys
    frase = " ".join(sys.argv[1:]) or "Não consigo jogar o Premier, levei ban VAC dos CTs?"
    print(frase)
    print(normalizar(frase))
