"""
lexer_afd.py - analisador léxico por Máquina de Estados Finitos (AFD).

Este módulo é uma adaptação, para Python e para texto em português, do
`analisador_lexico.c` da Aula 02 (Construção de Interpretadores, prof. Felipe
Przysiada). A ideia é a mesma do código do professor:

  PARTE 1 - o AFD de números (estados Q_INICIO ... Q_EXP_DIGITO), com a
            mesma função de transição delta(). Única mudança: a vírgula
            também é aceita como separador decimal (1,5 / 14,99).
  PARTE 2 - um segundo AFD, para palavras (letras, dígitos e hífen interno),
            no lugar dos operadores da calculadora.
  PARTE 3 - o lexer: pula espaços, tenta os AFDs a partir da posição atual
            e fica com o MAIOR prefixo aceito (maximal munch), igual ao
            reconhece_numero() do C.

Diferença importante em relação ao C: lá o lexer para no primeiro símbolo
fora do alfabeto (<INVALIDO>). Um chatbot não pode travar com um emoji, então
aqui o símbolo vira um token INVALIDO e a análise continua.

No projeto ele é usado de duas formas:
  1) como tokenizador alternativo ao NLTK (comparado no experimento de
     ablação), e
  2) para classificar cada token (SIGLA, NUMERO, PALAVRA...), o que permite
     proteger siglas como VAC, VRS e MR12 do stemming e das stopwords.
"""

import string
from dataclasses import dataclass
from enum import Enum, auto


# =============================================================
# PARTE 1 - AFD DE NÚMEROS (mesma MEF do analisador_lexico.c)
# =============================================================
#   MEF = { Q, Σ, δ, q0, A }
#   Q  = { INICIO, SINAL, INTEIRO, PONTO, DECIMAL,
#          EXP_INICIO, EXP_SINAL, EXP_DIGITO, ERRO }
#   Σ  = { '0'..'9', '.', ',', '+', '-', 'e', 'E' }
#   q0 = INICIO
#   A  = { INTEIRO, DECIMAL, EXP_DIGITO }
class EstadoNum(Enum):
    Q_INICIO = auto()
    Q_SINAL = auto()
    Q_INTEIRO = auto()      # aceita
    Q_PONTO = auto()
    Q_DECIMAL = auto()      # aceita
    Q_EXP_INICIO = auto()
    Q_EXP_SINAL = auto()
    Q_EXP_DIGITO = auto()   # aceita
    Q_ERRO = auto()


DIGITOS = "0123456789"
ACEITA_NUM = {EstadoNum.Q_INTEIRO, EstadoNum.Q_DECIMAL, EstadoNum.Q_EXP_DIGITO}


def delta_num(estado, c):
    """δ: Q x Σ -> Q  (função de transição do AFD de números)."""
    digito = c in DIGITOS
    sinal = c in "+-"
    expo = c in "eE"
    separador = c in ".,"   # adaptação: vírgula decimal (padrão brasileiro)

    if estado == EstadoNum.Q_INICIO:
        if sinal:
            return EstadoNum.Q_SINAL
        if digito:
            return EstadoNum.Q_INTEIRO
    elif estado == EstadoNum.Q_SINAL:
        if digito:
            return EstadoNum.Q_INTEIRO
    elif estado == EstadoNum.Q_INTEIRO:
        if digito:
            return EstadoNum.Q_INTEIRO
        if separador:
            return EstadoNum.Q_PONTO
        if expo:
            return EstadoNum.Q_EXP_INICIO
    elif estado == EstadoNum.Q_PONTO:
        if digito:
            return EstadoNum.Q_DECIMAL
    elif estado == EstadoNum.Q_DECIMAL:
        if digito:
            return EstadoNum.Q_DECIMAL
        if separador:  # 16.000,50 -> continua sendo número
            return EstadoNum.Q_PONTO
        if expo:
            return EstadoNum.Q_EXP_INICIO
    elif estado == EstadoNum.Q_EXP_INICIO:
        if sinal:
            return EstadoNum.Q_EXP_SINAL
        if digito:
            return EstadoNum.Q_EXP_DIGITO
    elif estado == EstadoNum.Q_EXP_SINAL:
        if digito:
            return EstadoNum.Q_EXP_DIGITO
    elif estado == EstadoNum.Q_EXP_DIGITO:
        if digito:
            return EstadoNum.Q_EXP_DIGITO
    return EstadoNum.Q_ERRO


def reconhece_numero(texto, pos):
    """Comprimento do MAIOR prefixo de texto[pos:] aceito pelo AFD de números
    (0 se nenhum número começa ali). Tradução direta da função do C."""
    estado = EstadoNum.Q_INICIO
    i = pos
    ultimo_aceito = -1
    while i < len(texto):
        proximo = delta_num(estado, texto[i])
        if proximo == EstadoNum.Q_ERRO:
            break  # o AFD "trava": para de processar
        estado = proximo
        i += 1
        if estado in ACEITA_NUM:
            ultimo_aceito = i
    if ultimo_aceito == -1:
        return 0
    return ultimo_aceito - pos


# =============================================================
# PARTE 2 - AFD DE PALAVRAS
# =============================================================
#   Q  = { INICIO, PALAVRA, HIFEN, ERRO }      A = { PALAVRA }
#   INICIO  --letra|dígito--> PALAVRA
#   PALAVRA --letra|dígito--> PALAVRA
#   PALAVRA --'-'-----------> HIFEN --letra|dígito--> PALAVRA
# O hífen só é aceito no meio ("pick-ban", "e-mail"); "cs-" para no "cs".
class EstadoPal(Enum):
    Q_INICIO = auto()
    Q_PALAVRA = auto()   # aceita
    Q_HIFEN = auto()
    Q_ERRO = auto()


def _letra_ou_digito(c):
    return c.isalpha() or c in DIGITOS


def delta_pal(estado, c):
    if estado in (EstadoPal.Q_INICIO, EstadoPal.Q_PALAVRA, EstadoPal.Q_HIFEN):
        if _letra_ou_digito(c):
            return EstadoPal.Q_PALAVRA
        if estado == EstadoPal.Q_PALAVRA and c == "-":
            return EstadoPal.Q_HIFEN
    return EstadoPal.Q_ERRO


def reconhece_palavra(texto, pos):
    estado = EstadoPal.Q_INICIO
    i = pos
    ultimo_aceito = -1
    while i < len(texto):
        proximo = delta_pal(estado, texto[i])
        if proximo == EstadoPal.Q_ERRO:
            break
        estado = proximo
        i += 1
        if estado == EstadoPal.Q_PALAVRA:
            ultimo_aceito = i
    if ultimo_aceito == -1:
        return 0
    return ultimo_aceito - pos


# =============================================================
# PARTE 3 - LEXER
# =============================================================
class TipoToken(Enum):
    PALAVRA = auto()
    SIGLA = auto()      # VAC, CS2, MR12, CTs ...
    NUMERO = auto()
    PONTUACAO = auto()
    INVALIDO = auto()   # fora do alfabeto (emoji, símbolos raros)


@dataclass
class Token:
    tipo: TipoToken
    lexema: str


PONTUACAO = set(string.punctuation) | set("¿¡«»“”‘’–—…´`ºª°")


def eh_sigla(lexema):
    """Sigla = só maiúsculas/dígitos, com pelo menos 2 caracteres e uma letra.
    Aceita plural com 's' minúsculo no final (CTs, TRs)."""
    corpo = lexema[:-1] if len(lexema) > 2 and lexema.endswith("s") else lexema
    if len(corpo) < 2 or not any(c.isalpha() for c in corpo):
        return False
    return all(c.isupper() or c in DIGITOS for c in corpo)


def classificar(lexema):
    """Roda os AFDs sobre um token já separado (usado também quando o
    tokenizador é o do NLTK) e devolve o tipo."""
    if not lexema:
        return TipoToken.INVALIDO
    if reconhece_numero(lexema, 0) == len(lexema):
        return TipoToken.NUMERO
    if reconhece_palavra(lexema, 0) == len(lexema):
        return TipoToken.SIGLA if eh_sigla(lexema) else TipoToken.PALAVRA
    if all(c in PONTUACAO for c in lexema):
        return TipoToken.PONTUACAO
    return TipoToken.INVALIDO


def tokenizar(texto):
    """Transforma a string em uma lista de Token, usando maior prefixo."""
    tokens = []
    pos = 0
    espera_numero = True  # mesma ideia do C: no início um sinal faz parte do número
    while pos < len(texto):
        c = texto[pos]
        if c.isspace():
            pos += 1
            espera_numero = True
            continue

        len_num = 0
        if c in DIGITOS or (c in "+-" and espera_numero):
            len_num = reconhece_numero(texto, pos)
        len_pal = reconhece_palavra(texto, pos) if _letra_ou_digito(c) else 0

        if len_num == 0 and len_pal == 0:
            tipo = TipoToken.PONTUACAO if c in PONTUACAO else TipoToken.INVALIDO
            tokens.append(Token(tipo, c))
            pos += 1
            espera_numero = True
            continue

        # maior prefixo: "2v2" vira palavra, "16.000" vira número
        if len_num >= len_pal:
            lexema = texto[pos:pos + len_num]
            tokens.append(Token(TipoToken.NUMERO, lexema))
        else:
            lexema = texto[pos:pos + len_pal]
            tipo = TipoToken.SIGLA if eh_sigla(lexema) else TipoToken.PALAVRA
            tokens.append(Token(tipo, lexema))
        pos += len(lexema)
        espera_numero = False
    return tokens


if __name__ == "__main__":
    # Mesmos casos do testar_afd() do analisador_lexico.c
    for caso in ["123", "-45.67", "3.14e-10", "abc", "12.", ".5", "1e10", "7"]:
        n = reconhece_numero(caso, 0)
        if n > 0 and n == len(caso):
            print(f'  "{caso}" -> ACEITA')
        elif n > 0:
            print(f'  "{caso}" -> só os primeiros {n} caractere(s) formam um número')
        else:
            print(f'  "{caso}" -> REJEITADA')
    print()
    for frase in ["Quanto custa o Prime? US$ 14,99 no CS2 😀", "pick-ban no MR12 dos CTs"]:
        print(frase)
        print("  ", [(t.tipo.name, t.lexema) for t in tokenizar(frase)])
