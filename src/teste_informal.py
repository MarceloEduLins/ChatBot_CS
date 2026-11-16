"""
teste_informal.py - a mesma pergunta escrita de três jeitos, para medir se o assistente
entende linguagem informal.

    python src/teste_informal.py

Para cada uma das 12 dúvidas abaixo há três versões da MESMA frase:
  formal      -> português padrão, com acentos;
  sem_acento  -> a mesma frase sem nenhum acento;
  abreviada   -> a mesma frase sem acento e com abreviações de chat (vc, pq, q, qnto, pra, tb...).
Como só muda o jeito de escrever, a diferença de acerto entre as colunas mede o efeito
dos acentos e das abreviações. Este conjunto não foi usado para escolher nada (limiar,
modelo, pipeline): é só uma verificação extra, feita depois da avaliação principal.

Cada versão roda duas vezes, sem e com o dicionário de abreviações
(normalizacao.ABREVIACOES), para medir o efeito dele.

Saída: resultados/teste_informal.csv e um resumo no terminal.
"""

import sys

import pandas as pd

from base import RAIZ, carregar_base
from chat import criar_assistente

# (id esperado, formal, sem acento, abreviada)
PERGUNTAS = [
    (41, "Quanto é que custa o Prime?", "Quanto e que custa o Prime?", "qnto eh q custa o prime?"),
    (40, "Como você compra o Prime?", "Como voce compra o Prime?", "como vc compra o prime?"),
    (34, "Por que minha patente não aparece mais?", "Por que minha patente nao aparece mais?",
     "pq minha patente n aparece mais?"),
    (81, "Você consegue ver o seu Trust Factor?", "Voce consegue ver o seu Trust Factor?",
     "vc consegue ver seu trust factor?"),
    (50, "Qual é o máximo de dinheiro que dá para juntar?",
     "Qual e o maximo de dinheiro que da para juntar?", "ql eh o maximo de dinheiro q da pra juntar?"),
    (56, "Quantos segundos até a bomba explodir?", "Quantos segundos ate a bomba explodir?",
     "qntos seg ate a bomba explodir?"),
    (14, "Quantas vitórias são necessárias para ter o CS Rating?",
     "Quantas vitorias sao necessarias para ter o CS Rating?",
     "qntas vitorias sao necessarias pra ter o cs rating?"),
    (85, "O CS2 também funciona no Mac?", "O CS2 tambem funciona no Mac?", "o cs2 tb funciona no mac?"),
    (90, "É possível pedir reembolso de uma caixa?", "E possivel pedir reembolso de uma caixa?",
     "eh possivel pedir reembolso d uma caixa?"),
    (72, "Dá para recorrer de um banimento do VAC?", "Da para recorrer de um banimento do VAC?",
     "da pra recorrer de um ban do vac?"),
    (28, "Uma partida competitiva tem quantas rodadas no máximo?",
     "Uma partida competitiva tem quantas rodadas no maximo?",
     "uma partida competitiva tem qntas rodadas no max?"),
    (87, "Como faço a verificação dos arquivos do jogo?", "Como faco a verificacao dos arquivos do jogo?",
     "como faco a verificacao dos arqvs do jogo?"),
]
FORMAS = ["formal", "sem_acento", "abreviada"]


def main():
    base = carregar_base()
    versoes = sys.argv[1:] or ["v1", "v2"]
    # cada versão roda duas vezes: sem e com o dicionário de abreviações
    assistentes = {}
    for v in versoes:
        assistentes[f"{v} sem dic"] = criar_assistente(v, base, expandir=False)
        assistentes[f"{v} com dic"] = criar_assistente(v, base, expandir=True)

    linhas = []
    for id_esperado, *textos in PERGUNTAS:
        esperado = base.canonico(id_esperado)
        for forma, texto in zip(FORMAS, textos):
            linha = {"id_esperado": esperado, "forma": forma, "pergunta": texto}
            for nome, a in assistentes.items():
                r = a.responder(texto)
                top1 = r.ranking[0][0] if r.ranking else -1
                col = nome.replace(" ", "_")
                linha[f"{col}_top1"] = top1
                linha[f"{col}_escore"] = round(r.escore if r.ranking else 0.0, 3)
                linha[f"{col}_ranking_certo"] = top1 == esperado
                linha[f"{col}_resposta_final_certa"] = r.id == esperado
            if "v1 com dic" in assistentes:
                linha["v1_termos_sem_dic"] = " · ".join(assistentes["v1 sem dic"].normalizador.normalizar(texto))
                linha["v1_termos_com_dic"] = " · ".join(assistentes["v1 com dic"].normalizador.normalizar(texto))
            linhas.append(linha)

    df = pd.DataFrame(linhas)
    saida = RAIZ / "resultados" / "teste_informal.csv"
    df.to_csv(saida, index=False, encoding="utf-8")

    print(f"\nRespostas finais certas (já com o limiar), em {len(PERGUNTAS)} perguntas por forma:\n")
    resumo = []
    for forma in FORMAS:
        parte = df[df.forma == forma]
        item = {"forma": forma}
        for nome in assistentes:
            item[nome] = int(parte[f"{nome.replace(' ', '_')}_resposta_final_certa"].sum())
        resumo.append(item)
    print(pd.DataFrame(resumo).to_string(index=False))
    print(f"\nDetalhe por pergunta (inclui o 1º lugar do ranking) em {saida}")


if __name__ == "__main__":
    main()
