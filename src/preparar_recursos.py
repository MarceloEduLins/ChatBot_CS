"""
preparar_recursos.py - baixa tudo que o projeto precisa, uma vez só.

    python src/preparar_recursos.py            # NLTK + modelo padrão da v2
    python src/preparar_recursos.py --tudo     # + modelo alternativo (~1 GB)
"""

import argparse

import nltk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tudo", action="store_true", help="baixa também o modelo alternativo")
    parser.add_argument("--so-nltk", action="store_true")
    args = parser.parse_args()

    for recurso in ["punkt_tab", "stopwords", "rslp"]:
        print(f"NLTK: {recurso}")
        nltk.download(recurso, quiet=True)

    if args.so_nltk:
        return
    from chatbot_denso import MODELO_ALTERNATIVO, MODELO_PADRAO, carregar_modelo
    modelos = [MODELO_PADRAO] + ([MODELO_ALTERNATIVO] if args.tudo else [])
    for nome in modelos:
        print(f"Modelo: {nome}")
        carregar_modelo(nome)
    print("Pronto.")


if __name__ == "__main__":
    main()
