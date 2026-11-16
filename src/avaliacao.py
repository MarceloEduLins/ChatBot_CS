"""
avaliacao.py - avaliação experimental da v1 (TF-IDF) e da v2 (embeddings).

Uso:
    python src/avaliacao.py                 # tudo (v1, v2, ablação, modelo alternativo)
    python src/avaliacao.py --sem-denso     # só v1 (não precisa de torch)
    python src/avaliacao.py --sem-alternativo

Regra de ouro: TODA escolha (campo indexado, normalizar ou não na v2, modelo,
limiar) é feita no conjunto de VALIDAÇÃO. O conjunto de TESTE só é usado no
fim, para medir. Assim o teste não "vaza" para as decisões.

Métricas (Seção 7 do enunciado):
  Acurácia@1       primeiro item do ranking == id esperado     (perguntas do escopo)
  MRR              média de 1/posição do id esperado          (perguntas do escopo)
  Rejeição correta fora do escopo que o assistente rejeitou    (perguntas fora do escopo)
  Falsa rejeição   do escopo rejeitadas pelo limiar            (perguntas do escopo)
Extra: acerto_final = resposta final certa (do escopo: id certo e aceito;
fora do escopo: rejeitada) sobre o total.

Saídas: resultados/*.csv, resultados/limiares.json, resultados/ambiente.json,
docs/resultados_experimentos.md e as tabelas automáticas de docs/decisoes.md.
"""

import argparse
import json
import platform
import random
import re
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from base import RAIZ, carregar_base, carregar_perguntas, checar_contaminacao
from chatbot_tfidf import AssistenteTFIDF
from normalizacao import NEGACOES, ConfigNormalizacao, Normalizador

RESULTADOS = RAIZ / "resultados"
DOCS = RAIZ / "docs"
GRADE_LIMIAR = np.round(np.arange(0.0, 1.0001, 0.01), 2)
SEMENTE = 42


def fixar_sementes(semente=SEMENTE):
    random.seed(semente)
    np.random.seed(semente)
    try:
        import torch
        torch.manual_seed(semente)
    except ImportError:
        pass


# ------------------------------------------------------------------
# Rodar um assistente sobre um conjunto de perguntas
# ------------------------------------------------------------------
def rodar(assistente, perguntas, base):
    """Uma passada por pergunta. Guardamos o ranking completo e o escore do
    1º colocado; o limiar é aplicado depois (assim dá para varrer limiares
    sem rodar o modelo de novo)."""
    linhas = []
    for pergunta, esperado in zip(perguntas["pergunta"], perguntas["id_esperado"]):
        r = assistente.responder(pergunta)
        esperado = base.canonico(esperado)
        ids = [i for i, _ in r.ranking]
        posicao = ids.index(esperado) + 1 if esperado in ids else None
        linhas.append({
            "pergunta": pergunta,
            "esperado": esperado,
            "top1": ids[0] if ids else -1,
            "escore": r.ranking[0][1] if r.ranking else 0.0,
            "posicao": posicao,
            "top3": ids[:3],
        })
    return linhas


def metricas(linhas, limiar):
    escopo = [l for l in linhas if l["esperado"] != -1]
    fora = [l for l in linhas if l["esperado"] == -1]
    media = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    acerto_aceito = [l["top1"] == l["esperado"] and l["escore"] >= limiar for l in escopo]
    rejeitou_fora = [l["escore"] < limiar for l in fora]
    return {
        "acuracia@1": media([l["top1"] == l["esperado"] for l in escopo]),
        "mrr": media([1.0 / l["posicao"] if l["posicao"] else 0.0 for l in escopo]),
        "rejeicao_correta": media(rejeitou_fora),
        "falsa_rejeicao": media([l["escore"] < limiar for l in escopo]),
        "acerto_final": media(acerto_aceito + rejeitou_fora),
        "n_escopo": len(escopo),
        "n_fora": len(fora),
    }


def escolher_limiar(linhas_validacao):
    """Varre o limiar na VALIDAÇÃO e maximiza
        J = 0,5 * (escopo respondido certo) + 0,5 * (fora do escopo rejeitado)
    O peso igual evita que as 24 perguntas do escopo "engulam" as 8 de fora.
    Empate: pega a maior faixa contínua de limiares empatados e usa o limiar
    do meio dela (o mais longe das bordas, menos sensível a pequenas mudanças)."""
    curva = []
    for t in GRADE_LIMIAR:
        m = metricas(linhas_validacao, t)
        escopo = [l for l in linhas_validacao if l["esperado"] != -1]
        acerto_aceito = np.mean([l["top1"] == l["esperado"] and l["escore"] >= t for l in escopo])
        j = 0.5 * acerto_aceito + 0.5 * m["rejeicao_correta"]
        curva.append({"limiar": t, "escopo_respondido_certo": acerto_aceito,
                      "rejeicao_correta": m["rejeicao_correta"],
                      "falsa_rejeicao": m["falsa_rejeicao"], "J": j})
    curva = pd.DataFrame(curva)
    empatados = np.isclose(curva["J"].to_numpy(), curva["J"].max())
    faixas, atual = [], []
    for t, ok in zip(curva["limiar"], empatados):
        if ok:
            atual.append(t)
        elif atual:
            faixas.append(atual)
            atual = []
    if atual:
        faixas.append(atual)
    maior = max(faixas, key=len)          # em caso de mesmo tamanho, fica a primeira (menor limiar)
    limiar = float(maior[(len(maior) - 1) // 2])
    return limiar, curva


def chave_selecao(m):
    """Critério para escolher configurações na validação: MRR, depois acurácia."""
    return (round(m["mrr"], 6), round(m["acuracia@1"], 6))


# ------------------------------------------------------------------
# Tabelas em Markdown (sem depender do pacote tabulate)
# ------------------------------------------------------------------
def tabela_md(df, casas=3):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:.{casas}f}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    corpo = ["| " + " | ".join(fmt(v) for v in linha) + " |" for linha in df.itertuples(index=False)]
    return "\n".join([cab, sep] + corpo)


def preencher_marcador(caminho, nome, conteudo):
    """Troca o texto entre <!-- AUTO:nome --> e <!-- /AUTO:nome --> no .md."""
    texto = caminho.read_text(encoding="utf-8")
    padrao = re.compile(rf"(<!-- AUTO:{nome} -->)(.*?)(<!-- /AUTO:{nome} -->)", re.S)
    if not padrao.search(texto):
        print(f"[aviso] marcador AUTO:{nome} não encontrado em {caminho.name}")
        return
    texto = padrao.sub(lambda m: f"{m.group(1)}\n{conteudo}\n{m.group(3)}", texto)
    caminho.write_text(texto, encoding="utf-8")


# ------------------------------------------------------------------
# v0: quadro antes/depois
# ------------------------------------------------------------------
IDS_QUADRO = [73, 100, 16, 43, 85]   # negação, sigla, sigla+acento, "sem"/"sumiu", pergunta curta
EXTRA_QUADRO = ["não consigo entrar no premier, o botão tá cinza",
                "qnto custa o prime? pq n consigo comprar"]


def quadro_v0(base):
    padrao = Normalizador(ConfigNormalizacao(), siglas=base.siglas)
    so_token = Normalizador(ConfigNormalizacao(minusculas=False, remover_pontuacao=False,
                                               remover_stopwords=False, stemming=False,
                                               remover_acentos=False, preservar_siglas=False,
                                               expandir_abreviacoes=False))
    ingenuo = Normalizador(ConfigNormalizacao(manter_negacao=False, preservar_siglas=False,
                                              expandir_abreviacoes=False),
                           siglas=base.siglas)
    linhas = []
    textos = [(f"id {i}", base.pergunta(i)) for i in IDS_QUADRO]
    textos += [("extra (fora da base)", t) for t in EXTRA_QUADRO]
    for origem, texto in textos:
        linhas.append({
            "origem": origem,
            "antes (texto original)": texto,
            "só tokenização": " · ".join(so_token.normalizar(texto)),
            "pipeline ingênuo (stopwords NLTK puras, sem siglas, sem abreviações)": " · ".join(ingenuo.normalizar(texto)),
            "pipeline adotado": " · ".join(padrao.normalizar(texto)),
        })
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------
# Análise de erros automática (hipóteses iniciais)
# ------------------------------------------------------------------
def hipoteses(pergunta, esperado, previsto, base, normalizador, vocab, campo):
    """Pistas automáticas para a análise de erros. São só um ponto de partida:
    a explicação final de cada caso é escrita pelo grupo em docs/decisoes.md."""
    textos = dict(zip(base.ids, base.textos(campo)))

    def termos_do_grupo(id_):
        membros = [i for i in base.ids if base.grupo_de[i] == id_]
        return set(t for i in membros for t in normalizador.normalizar(textos[i]))

    tq = normalizador.normalizar(pergunta)
    te = termos_do_grupo(esperado) if esperado != -1 else set()
    comuns = set(tq) & te
    pistas = []
    if not tq:
        pistas.append("tudo virou stopword/pontuação")
    if esperado != -1 and not comuns:
        pistas.append(f"nenhum termo em comum com o texto indexado ({campo}) do id esperado "
                      "(sinonímia/vocabulário)")
    fora_vocab = [t for t in tq if t not in vocab]
    if fora_vocab:
        pistas.append("fora do vocabulário TF-IDF: " + ", ".join(fora_vocab[:4]))
    if any(t in base.siglas for t in tq):
        pistas.append("tem sigla: " + ", ".join(t for t in tq if t in base.siglas))
    if any(p.lower().strip("?,.!") in NEGACOES for p in pergunta.split()):
        pistas.append("tem negação")
    if 0 < len(tq) <= 2:
        pistas.append(f"pergunta curta ({len(tq)} termos após v0)")
    if previsto not in (-1, esperado):
        enganou = sorted(set(tq) & termos_do_grupo(previsto) - comuns)
        if enganou:
            pistas.append(f"termos que puxaram para o id {previsto}: " + ", ".join(enganou[:4]))
    return "; ".join(pistas) or "-"


# ------------------------------------------------------------------
# Principal
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sem-denso", action="store_true", help="não roda a v2 (sem torch)")
    parser.add_argument("--sem-alternativo", action="store_true",
                        help="não compara com o modelo alternativo (download maior)")
    args = parser.parse_args()

    inicio = time.time()
    fixar_sementes()
    RESULTADOS.mkdir(exist_ok=True)
    base = carregar_base()
    val = carregar_perguntas("validacao")
    teste = carregar_perguntas("teste")
    md = ["# Resultados dos experimentos (gerado por src/avaliacao.py)", "",
          f"Base: {len(base)} pares ({len(set(base.grupo_de.values()))} respostas distintas). "
          f"Validação: {len(val)} perguntas. Teste: {len(teste)} perguntas. Semente: {SEMENTE}.", ""]

    # 0) checagens de contaminação
    for nome, df in [("validacao", val), ("teste", teste)]:
        copiadas = checar_contaminacao(base, df)
        print(f"[{nome}] perguntas copiadas literalmente da base: {len(copiadas)}")
        md.append(f"- Perguntas de `{nome}` copiadas literalmente da base: **{len(copiadas)}** {copiadas or ''}")
    md.append("")

    # 1) quadro v0
    quadro = quadro_v0(base)
    quadro.to_csv(RESULTADOS / "quadro_v0.csv", index=False)
    preencher_marcador(DOCS / "decisoes.md", "quadro_v0", tabela_md(quadro))

    # 2) v1: o que indexar (escolhido na validação)
    print("v1: experimento de indexação...")
    exp_campo = []
    for campo in ["pergunta", "resposta", "pergunta+resposta"]:
        a = AssistenteTFIDF(base, campo=campo)
        lv = rodar(a, val, base)
        limiar, _ = escolher_limiar(lv)
        mv = metricas(lv, limiar)
        mt = metricas(rodar(a, teste, base), limiar)
        exp_campo.append({"versao": "v1", "campo": campo, "limiar_val": limiar,
                          "val_acuracia@1": mv["acuracia@1"], "val_mrr": mv["mrr"],
                          "teste_acuracia@1": mt["acuracia@1"], "teste_mrr": mt["mrr"]})
    campo_v1 = max(exp_campo, key=lambda e: (round(e["val_mrr"], 6), round(e["val_acuracia@1"], 6)))["campo"]

    # 3) v1 final
    v1 = AssistenteTFIDF(base, campo=campo_v1)
    lv1_val = rodar(v1, val, base)
    limiar_v1, curva_v1 = escolher_limiar(lv1_val)
    curva_v1.to_csv(RESULTADOS / "curva_limiar_v1_validacao.csv", index=False)
    lv1_teste = rodar(v1, teste, base)
    m_v1 = metricas(lv1_teste, limiar_v1)
    vocab_v1 = set(v1.vocabulario())

    # 4) ablação do pipeline v0 sobre a v1
    print("v1: ablação do pipeline...")
    variantes = {
        "pipeline adotado": {},
        "sem expandir abreviações": {"expandir_abreviacoes": False},
        "sem remover stopwords": {"remover_stopwords": False},
        "stopwords NLTK puras (remove 'não')": {"manter_negacao": False},
        "+ marcar negação (nao_)": {"marcar_negacao": True},
        "+ manter interrogativos": {"manter_interrogativos": True},
        "sem stemming": {"stemming": False},
        "sem remover acentos": {"remover_acentos": False},
        "sem proteger siglas": {"preservar_siglas": False},
        "tokenizador AFD (lexer da Aula 02)": {"tokenizador": "afd"},
    }
    ablacao = []
    for nome, mudanca in variantes.items():
        cfg = replace(ConfigNormalizacao(), **mudanca)
        a = AssistenteTFIDF(base, campo=campo_v1, config=cfg)
        lv = rodar(a, val, base)
        lim, _ = escolher_limiar(lv)
        mv, mt = metricas(lv, lim), metricas(rodar(a, teste, base), lim)
        ablacao.append({"variante": nome, "limiar_val": lim,
                        "val_acuracia@1": mv["acuracia@1"], "val_mrr": mv["mrr"],
                        "teste_acuracia@1": mt["acuracia@1"], "teste_mrr": mt["mrr"],
                        "teste_rejeicao_correta": mt["rejeicao_correta"],
                        "teste_falsa_rejeicao": mt["falsa_rejeicao"],
                        "vocabulario": len(a.vocabulario())})
    ablacao = pd.DataFrame(ablacao)
    ablacao.to_csv(RESULTADOS / "ablacao_v1.csv", index=False)

    comparacao = [{"versao": "v1", "representacao": "TF-IDF", "campo": campo_v1,
                   "limiar": limiar_v1, **{k: m_v1[k] for k in
                   ["acuracia@1", "mrr", "rejeicao_correta", "falsa_rejeicao", "acerto_final"]}}]
    limiares = {"v1": {"limiar": limiar_v1, "campo": campo_v1, "expandir_abreviacoes": True}}
    ambiente = {"python": sys.version.split()[0], "plataforma": platform.platform(),
                "semente": SEMENTE}
    for pacote in ["nltk", "sklearn", "numpy", "pandas"]:
        ambiente[pacote] = __import__(pacote).__version__

    # 5) v2
    detalhe = pd.DataFrame({"pergunta": teste["pergunta"],
                            "esperado": [base.canonico(i) for i in teste["id_esperado"]]})
    detalhe["v1_top1"] = [l["top1"] for l in lv1_teste]
    detalhe["v1_escore"] = [l["escore"] for l in lv1_teste]
    detalhe["v1_posicao"] = [l["posicao"] for l in lv1_teste]
    detalhe["v1_final"] = [l["top1"] if l["escore"] >= limiar_v1 else -1 for l in lv1_teste]

    exp_denso, m_v2 = [], None
    if not args.sem_denso:
        try:
            from chatbot_denso import (MODELO_ALTERNATIVO, MODELO_PADRAO,
                                       AssistenteDenso, info_modelo)
        except ImportError as erro:
            print(f"[aviso] v2 não rodou ({erro}). Instale o requirements.txt.")
            args.sem_denso = True

    if not args.sem_denso:
        print("v2: experimentos (indexação, normalizar ou não, modelo)...")

        def avaliar_denso(etapa, **kw):
            a = AssistenteDenso(base, **kw)
            lv = rodar(a, val, base)
            lim, _ = escolher_limiar(lv)
            mv, mt = metricas(lv, lim), metricas(rodar(a, teste, base), lim)
            exp_denso.append({"etapa": etapa, "modelo": kw["modelo"].split("/")[-1],
                              "campo": kw["campo"], "normalizar_antes": kw["normalizar"],
                              "expandir_abreviacoes": kw.get("expandir_abreviacoes", True),
                              "limiar_val": lim,
                              "val_acuracia@1": mv["acuracia@1"], "val_mrr": mv["mrr"],
                              "teste_acuracia@1": mt["acuracia@1"], "teste_mrr": mt["mrr"]})
            return mv

        # 5a) campo, com o modelo padrão e texto cru
        melhor = None
        for campo in ["pergunta", "resposta", "pergunta+resposta"]:
            mv = avaliar_denso("campo", modelo=MODELO_PADRAO, campo=campo, normalizar=False)
            if melhor is None or chave_selecao(mv) > melhor[0]:
                melhor = (chave_selecao(mv), campo)
        campo_v2 = melhor[1]
        # 5b) normalizar antes do embedding?
        m_cru = [e for e in exp_denso if e["etapa"] == "campo" and e["campo"] == campo_v2][0]
        mv_norm = avaliar_denso("normalizacao", modelo=MODELO_PADRAO, campo=campo_v2, normalizar=True)
        normalizar_v2 = chave_selecao(mv_norm) > (round(m_cru["val_mrr"], 6), round(m_cru["val_acuracia@1"], 6))
        # 5c) modelo alternativo
        modelo_v2 = MODELO_PADRAO
        if not args.sem_alternativo:
            mv_alt = avaliar_denso("modelo", modelo=MODELO_ALTERNATIVO, campo=campo_v2,
                                   normalizar=normalizar_v2)
            ref = [e for e in exp_denso if e["modelo"] == MODELO_PADRAO.split("/")[-1]
                   and e["campo"] == campo_v2 and e["normalizar_antes"] == normalizar_v2][0]
            if chave_selecao(mv_alt) > (round(ref["val_mrr"], 6), round(ref["val_acuracia@1"], 6)):
                modelo_v2 = MODELO_ALTERNATIVO
        # 5d) expandir abreviações? Adotado (ideia tirada do enunciado: registro informal),
        #     a não ser que DESLIGAR seja melhor na validação.
        ref = [e for e in exp_denso if e["modelo"] == modelo_v2.split("/")[-1]
               and e["campo"] == campo_v2 and e["normalizar_antes"] == normalizar_v2
               and e["expandir_abreviacoes"]][0]
        mv_sem = avaliar_denso("abreviacoes", modelo=modelo_v2, campo=campo_v2,
                               normalizar=normalizar_v2, expandir_abreviacoes=False)
        expandir_v2 = not (chave_selecao(mv_sem) > (round(ref["val_mrr"], 6), round(ref["val_acuracia@1"], 6)))
        pd.DataFrame(exp_denso).to_csv(RESULTADOS / "experimentos_v2.csv", index=False)

        # 5e) v2 final
        v2 = AssistenteDenso(base, modelo=modelo_v2, campo=campo_v2, normalizar=normalizar_v2,
                             expandir_abreviacoes=expandir_v2)
        lv2_val = rodar(v2, val, base)
        limiar_v2, curva_v2 = escolher_limiar(lv2_val)
        curva_v2.to_csv(RESULTADOS / "curva_limiar_v2_validacao.csv", index=False)
        lv2_teste = rodar(v2, teste, base)
        m_v2 = metricas(lv2_teste, limiar_v2)
        comparacao.append({"versao": "v2", "representacao": f"embeddings ({modelo_v2.split('/')[-1]})",
                           "campo": campo_v2 + (" (normalizado)" if normalizar_v2 else ""),
                           "limiar": limiar_v2, **{k: m_v2[k] for k in
                           ["acuracia@1", "mrr", "rejeicao_correta", "falsa_rejeicao", "acerto_final"]}})
        limiares["v2"] = {"limiar": limiar_v2, "modelo": modelo_v2, "campo": campo_v2,
                          "normalizar": normalizar_v2, "expandir_abreviacoes": expandir_v2}
        ambiente["v2"] = info_modelo(modelo_v2)
        detalhe["v2_top1"] = [l["top1"] for l in lv2_teste]
        detalhe["v2_escore"] = [l["escore"] for l in lv2_teste]
        detalhe["v2_posicao"] = [l["posicao"] for l in lv2_teste]
        detalhe["v2_final"] = [l["top1"] if l["escore"] >= limiar_v2 else -1 for l in lv2_teste]

    # 6) salvar comparação, limiares e ambiente
    comparacao = pd.DataFrame(comparacao)
    comparacao.to_csv(RESULTADOS / "comparacao.csv", index=False)
    (RESULTADOS / "limiares.json").write_text(json.dumps(limiares, indent=2, ensure_ascii=False))
    (RESULTADOS / "ambiente.json").write_text(json.dumps(ambiente, indent=2, ensure_ascii=False))

    # 7) casos divergentes (só perguntas do escopo)
    tem_v2 = "v2_top1" in detalhe
    detalhe["dicas_hipotese"] = [
        hipoteses(p, e, t, base, v1.normalizador, vocab_v1, campo_v1)
        for p, e, t in zip(detalhe["pergunta"], detalhe["esperado"], detalhe["v1_top1"])]
    detalhe.to_csv(RESULTADOS / "detalhe_teste.csv", index=False)

    escopo = detalhe[detalhe["esperado"] != -1]
    if tem_v2:
        v2_ganha = escopo[(escopo["v2_top1"] == escopo["esperado"]) & (escopo["v1_top1"] != escopo["esperado"])]
        v1_ganha = escopo[(escopo["v1_top1"] == escopo["esperado"]) & (escopo["v2_top1"] != escopo["esperado"])]
        cols = ["pergunta", "esperado", "v1_top1", "v1_escore", "v2_top1", "v2_escore", "dicas_hipotese"]
        casos = (f"**v2 acerta e v1 erra ({len(v2_ganha)} casos)**\n\n"
                 + (tabela_md(v2_ganha[cols]) if len(v2_ganha) else "_nenhum caso_")
                 + f"\n\n**v1 acerta e v2 erra ({len(v1_ganha)} casos)**\n\n"
                 + (tabela_md(v1_ganha[cols]) if len(v1_ganha) else
                    "_nenhum caso: a v2 acertou tudo que a v1 acertou (ver discussão abaixo)_"))
        ambas_erram = escopo[(escopo["v1_top1"] != escopo["esperado"]) & (escopo["v2_top1"] != escopo["esperado"])]
        casos += (f"\n\n**As duas erram ({len(ambas_erram)} casos)**\n\n"
                  + (tabela_md(ambas_erram[cols]) if len(ambas_erram) else "_nenhum_"))
    else:
        erros = escopo[escopo["v1_top1"] != escopo["esperado"]]
        casos = "_v2 não foi executada._\n\n" + tabela_md(
            erros[["pergunta", "esperado", "v1_top1", "v1_escore", "dicas_hipotese"]])

    # 8) relatórios em markdown
    tab_comp = tabela_md(comparacao)
    tab_campo = tabela_md(pd.DataFrame(exp_campo))
    tab_abl = tabela_md(ablacao)
    pontos = curva_v1[np.isclose((curva_v1["limiar"] * 100) % 5, 0)]
    tab_curva_v1 = tabela_md(pontos)
    md += ["## Comparação v1 × v2 (conjunto de teste)", "", tab_comp, "",
           "## v1 – o que indexar (escolha pela validação)", "", tab_campo, "",
           f"Campo escolhido para a v1: **{campo_v1}**. Limiar escolhido na validação: **{limiar_v1:.2f}**.", "",
           "## v1 – curva do limiar na validação (a cada 0,05)", "", tab_curva_v1, "",
           "## Ablação do pipeline v0 (sobre a v1)", "", tab_abl, ""]
    if exp_denso:
        md += ["## v2 – experimentos (campo, normalização, modelo)", "",
               tabela_md(pd.DataFrame(exp_denso)), "",
               f"Escolhas da v2: campo **{limiares['v2']['campo']}**, normalizar antes: "
               f"**{limiares['v2']['normalizar']}**, modelo **{limiares['v2']['modelo']}**, "
               f"limiar **{limiares['v2']['limiar']:.2f}**.", "",
               "## v2 – curva do limiar na validação (a cada 0,05)", "",
               tabela_md(curva_v2[np.isclose((curva_v2["limiar"] * 100) % 5, 0)]), ""]
    md += ["## Casos para a análise de erros (teste, perguntas do escopo)", "", casos, "",
           f"_Tempo total: {time.time() - inicio:.1f} s._"]
    (DOCS / "resultados_experimentos.md").write_text("\n".join(md), encoding="utf-8")

    decisoes = DOCS / "decisoes.md"
    preencher_marcador(decisoes, "comparacao", tab_comp)
    preencher_marcador(decisoes, "indexacao", tab_campo + f"\n\nCampo escolhido: **{campo_v1}**.")
    preencher_marcador(decisoes, "ablacao", tab_abl)
    preencher_marcador(decisoes, "limiares", "```json\n" + json.dumps(limiares, indent=2, ensure_ascii=False) + "\n```")
    preencher_marcador(decisoes, "curva_v1", tab_curva_v1)
    if exp_denso:
        preencher_marcador(decisoes, "experimentos_v2", tabela_md(pd.DataFrame(exp_denso)))
    preencher_marcador(decisoes, "casos", casos)

    print()
    print(comparacao.to_string(index=False))
    print(f"\nResultados em {RESULTADOS} e {DOCS / 'resultados_experimentos.md'} "
          f"({time.time() - inicio:.1f} s)")


if __name__ == "__main__":
    main()
