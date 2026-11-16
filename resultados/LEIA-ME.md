Esta pasta é preenchida por `python src/avaliacao.py`:

- `comparacao.csv` – métricas de v1 e v2 no teste (Acurácia@1, MRR, rejeição correta, falsa rejeição, acerto_final)
- `ablacao_v1.csv` – efeito de ligar/desligar cada etapa do pipeline v0
- `experimentos_v2.csv` – campo indexado, normalizar ou não, modelo padrão × alternativo
- `curva_limiar_v1_validacao.csv`, `curva_limiar_v2_validacao.csv` – varredura do limiar na validação
- `detalhe_teste.csv` – resultado de cada pergunta do teste nas duas versões
- `quadro_v0.csv` – quadro antes/depois da normalização
- `limiares.json` – limiares e configurações escolhidos (lidos pelo chat.py)
- `ambiente.json` – versões das bibliotecas e do modelo
- `bonus_lora_dora.csv` e `bonus_exemplos.csv` – gerados por `python src/bonus_lora_dora.py`
- `teste_informal.csv` – mesma pergunta formal, sem acento e abreviada; gerado por `python src/teste_informal.py`
