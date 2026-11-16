"""
bonus_lora_dora.py - Bônus (Seção 4.4): ajuste fino eficiente com LoRA e DoRA.

Tarefa de demonstração: classificar a CATEGORIA de uma pergunta (premier,
punicoes, economia, ...), usando a coluna `categoria` de dados/base.csv.
  - treino: as 100 perguntas da base
  - avaliação: perguntas do escopo de validacao.csv + teste.csv (categoria
    tirada do id esperado). Nenhuma decisão é tomada olhando essa avaliação.

Modelo: o mesmo encoder da v2 (paraphrase-multilingual-MiniLM-L12-v2, já
baixado), com uma cabeça de classificação nova.

LoRA: o peso original W (congelado) ganha um desvio de posto baixo,
    W' = W + (alpha/r) · B·A,   A: r x k,  B: d x r,  r << d, k
Só A e B (e a cabeça de classificação) são treinados.
DoRA (use_dora=True): decompõe W em magnitude e direção,
    W' = m · (W + B·A) / ||W + B·A||
treinando o vetor de magnitude m além de A e B. Custa poucos parâmetros a
mais que a LoRA.

O enunciado não exige ganho de desempenho, e sim a demonstração funcional e
a contagem de parâmetros treináveis x total.

    python src/bonus_lora_dora.py            # ~2 a 5 min em CPU
    python src/bonus_lora_dora.py --pergunta "como denuncio um hacker?"

Além da tabela de parâmetros, o script mostra a categoria que cada método
prevê para algumas perguntas de exemplo (e para as passadas em --pergunta),
para dar para ver o modelo ajustado funcionando.
"""

import argparse
import time

import numpy as np
import pandas as pd

from base import RAIZ, carregar_base, carregar_perguntas
from chatbot_denso import MODELO_PADRAO

SEMENTE = 42
EPOCAS = 12
LOTE = 16
TAXA = 2e-3
R = 8
ALPHA = 16
ALVOS = ["query", "value"]   # projeções da atenção onde entram os adaptadores
METODOS = ["cabeça apenas", "LoRA", "DoRA"]
# perguntas novas (não estão na base) só para mostrar o classificador funcionando
EXEMPLOS = [
    "quanto custa o prime?",
    "levei ban por sair no meio da partida",
    "em quanto tempo a bomba explode?",
    "como peço reembolso de uma chave?",
    "quando vai ser o próximo major?",
]


def dados(base):
    categorias = sorted(base.df["categoria"].unique())
    rotulo = {c: i for i, c in enumerate(categorias)}
    treino_x = base.df["pergunta"].tolist()
    treino_y = [rotulo[c] for c in base.df["categoria"]]
    cat_do_id = dict(zip(base.df["id"], base.df["categoria"]))
    aval = pd.concat([carregar_perguntas("validacao"), carregar_perguntas("teste")])
    aval = aval[aval["id_esperado"] != -1]
    aval_x = aval["pergunta"].tolist()
    aval_y = [rotulo[cat_do_id[i]] for i in aval["id_esperado"]]
    return categorias, treino_x, treino_y, aval_x, aval_y


def treinar(metodo, base, exemplos=EXEMPLOS):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEMENTE)
    np.random.seed(SEMENTE)
    categorias, tx, ty, ax, ay = dados(base)
    tok = AutoTokenizer.from_pretrained(MODELO_PADRAO)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_PADRAO, num_labels=len(categorias))
    total_original = sum(p.numel() for p in modelo.parameters())

    if metodo == "cabeça apenas":
        for nome, p in modelo.named_parameters():
            p.requires_grad = nome.startswith("classifier")
        treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        total = total_original
    else:
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=R, lora_alpha=ALPHA, lora_dropout=0.1,
                         target_modules=ALVOS, use_dora=(metodo == "DoRA"))
        modelo = get_peft_model(modelo, cfg)
        treinaveis, total = modelo.get_nb_trainable_parameters()

    def prever(textos):
        """Índice da categoria mais provável para cada texto."""
        modelo.eval()
        with torch.no_grad():
            enc = tok(textos, padding=True, truncation=True, max_length=64, return_tensors="pt")
            pred = modelo(**enc).logits.argmax(-1).numpy()
        modelo.train()
        return pred

    def acuracia():
        return float(np.mean(prever(ax) == np.array(ay)))

    otimizador = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=TAXA)
    modelo.train()
    inicio = time.time()
    rng = np.random.default_rng(SEMENTE)
    for epoca in range(EPOCAS):
        ordem = rng.permutation(len(tx))
        perdas = []
        for i in range(0, len(ordem), LOTE):
            idx = ordem[i:i + LOTE]
            enc = tok([tx[j] for j in idx], padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
            saida = modelo(**enc, labels=torch.tensor([ty[j] for j in idx]))
            saida.loss.backward()
            otimizador.step()
            otimizador.zero_grad()
            perdas.append(saida.loss.item())
        print(f"  {metodo:14s} época {epoca + 1:2d}  perda={np.mean(perdas):.3f}")

    tempo = time.time() - inicio
    metricas = {"método": metodo, "parâmetros treináveis": int(treinaveis),
                "parâmetros totais": int(total),
                "% treinável": 100.0 * treinaveis / total,
                "acurácia (val+teste do escopo)": acuracia(),
                "tempo de treino (s)": tempo}
    previsoes = [categorias[i] for i in prever(list(exemplos))]
    return metricas, previsoes


def main():
    parser = argparse.ArgumentParser(description="Bônus: LoRA e DoRA com PEFT")
    parser.add_argument("--pergunta", action="append", default=[],
                        help="pergunta extra para classificar (pode repetir a opção)")
    args = parser.parse_args()
    exemplos = EXEMPLOS + args.pergunta

    base = carregar_base()
    linhas, colunas = [], {"pergunta": exemplos}
    for metodo in METODOS:
        print(f"Treinando: {metodo}")
        metricas, previsoes = treinar(metodo, base, exemplos)
        linhas.append(metricas)
        colunas[metodo] = previsoes
    df = pd.DataFrame(linhas)
    demo = pd.DataFrame(colunas)

    pasta = RAIZ / "resultados"
    pasta.mkdir(exist_ok=True)
    df.to_csv(pasta / "bonus_lora_dora.csv", index=False)
    demo.to_csv(pasta / "bonus_exemplos.csv", index=False)

    print("\nParâmetros treináveis x total:")
    print(df.to_string(index=False))
    print("\nCategoria prevista para perguntas novas:")
    print(demo.to_string(index=False))
    print(f"\nSalvo em {pasta}")

    # preenche a seção do bônus em docs/decisoes.md (se o marcador existir)
    from avaliacao import DOCS, preencher_marcador, tabela_md
    preencher_marcador(DOCS / "decisoes.md", "bonus",
                       tabela_md(df) + "\n\nCategoria prevista para perguntas que não estão na base:\n\n"
                       + tabela_md(demo))


if __name__ == "__main__":
    main()
