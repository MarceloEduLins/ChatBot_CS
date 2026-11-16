"""
chat.py - conversa com qualquer versão do assistente pelo terminal.

    python chat.py --versao v1
    python chat.py --versao v2
    python chat.py --versao ambas          (v1 e v2 lado a lado, bom para a demonstração)
    python chat.py --versao v1 --pergunta "quanto custa o prime?" --ranking 3

Os limiares (e o campo/modelo) vêm de resultados/limiares.json, gerado pelo
src/avaliacao.py. Se o arquivo ainda não existir, usa os valores padrão.
Digite 'sair' (ou Ctrl+D / Ctrl+C) para encerrar.
"""

import argparse
import json
import sys

from base import RAIZ, carregar_base


def criar_assistente(versao, base, limiar=None, expandir=None):
    """expandir=None usa o que a avaliação escolheu (limiares.json); True/False força."""
    config = {}
    arq = RAIZ / "resultados" / "limiares.json"
    if arq.exists():
        config = json.loads(arq.read_text(encoding="utf-8")).get(versao, {})
    if limiar is not None:
        config["limiar"] = limiar
    if expandir is None:
        expandir = config.get("expandir_abreviacoes", True)

    if versao == "v1":
        from chatbot_tfidf import LIMIAR_PADRAO_V1, AssistenteTFIDF
        from normalizacao import ConfigNormalizacao
        return AssistenteTFIDF(base, limiar=config.get("limiar", LIMIAR_PADRAO_V1),
                               campo=config.get("campo", "pergunta"),
                               config=ConfigNormalizacao(expandir_abreviacoes=expandir))
    from chatbot_denso import LIMIAR_PADRAO_V2, MODELO_PADRAO, AssistenteDenso
    return AssistenteDenso(base, limiar=config.get("limiar", LIMIAR_PADRAO_V2),
                           modelo=config.get("modelo", MODELO_PADRAO),
                           campo=config.get("campo", "pergunta"),
                           normalizar=config.get("normalizar", False),
                           expandir_abreviacoes=expandir)


def formatar(versao, r, base, n_ranking=0):
    if r.rejeitado:
        linha = f"[{versao}] {r.resposta} (escore={r.escore:.2f} < limiar {r.limiar:.2f})"
    else:
        linha = f"[{versao}] (id={r.id}, escore={r.escore:.2f}) {r.resposta}"
    for pos, (id_, s) in enumerate(r.ranking[:n_ranking], start=1):
        linha += f"\n      {pos}º id={id_:<3} {s:.3f}  {base.pergunta(id_)}"
    return linha


def main():
    parser = argparse.ArgumentParser(description="Chatbot FAQ de Counter-Strike 2")
    parser.add_argument("--versao", choices=["v1", "v2", "ambas"], default="v1",
                        help="'ambas' mostra a resposta da v1 e da v2 para a mesma pergunta")
    parser.add_argument("--limiar", type=float, default=None,
                        help="sobrescreve o limiar salvo (não vale para --versao ambas)")
    parser.add_argument("--ranking", type=int, default=0, help="mostra os N primeiros do ranking")
    parser.add_argument("--pergunta", default=None, help="responde uma pergunta e sai")
    args = parser.parse_args()

    base = carregar_base()
    if args.versao == "ambas":      # cada versão usa o seu próprio limiar
        versoes, limiar = ["v1", "v2"], None
    else:
        versoes, limiar = [args.versao], args.limiar
    print(f"Carregando {' e '.join(versoes)}...", file=sys.stderr)
    assistentes = {v: criar_assistente(v, base, limiar) for v in versoes}

    def responder(pergunta):
        for versao, assistente in assistentes.items():
            try:
                r = assistente.responder(pergunta)
                print(formatar(versao, r, base, args.ranking))
            except Exception as erro:  # o chat nunca pode cair
                print(f"[{versao}] Tive um problema com essa pergunta ({type(erro).__name__}). "
                      "Pode reformular?")

    if args.pergunta is not None:
        responder(args.pergunta)
        return

    print("FAQ de CS2 (competitivo e suporte). Digite 'sair' para encerrar.")
    while True:
        try:
            pergunta = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if pergunta.strip().lower() in {"sair", "exit", "quit"}:
            break
        responder(pergunta)


if __name__ == "__main__":
    main()
