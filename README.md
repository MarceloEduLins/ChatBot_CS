# Chatbot de FAQ – Counter-Strike 2 (PjBL 2)

Construção de Interpretadores – Ciência da Computação – PUCPR – 2026/2
Prof. Felipe A. Przysiada

**Dupla:** Marcelo Lins e Miguel Wihby

**Domínio:** perguntas frequentes sobre o **Counter-Strike 2**, cobrindo o **competitivo**
(Premier, CS Rating, patentes, regras de rodada, economia, Majors e ranking da Valve) e o
**suporte ao jogador** (Status Prime, cooldowns, VAC, Trust Factor, trocas, reembolso,
problemas técnicos).

O assistente foi implementado em três gerações que convivem no projeto:

| Versão | O que é | Arquivo |
|---|---|---|
| v0 | pipeline de normalização configurável (`normalizar(texto) -> list[str]`), com dicionário de abreviações de chat ("vc", "qnto", "n") | `src/normalizacao.py` (+ `src/lexer_afd.py`) |
| v1 | recuperação com TF-IDF + similaridade de cossenos própria + limiar | `src/chatbot_tfidf.py` (+ `src/similaridade.py`) |
| v2 | recuperação com sentence embeddings pré-treinados + limiar | `src/chatbot_denso.py` |
| bônus | ajuste fino eficiente com LoRA e DoRA (PEFT) | `src/bonus_lora_dora.py` |

Decisões de projeto, quadro antes/depois, ablação e análise de erros: **`docs/decisoes.md`**.

---

## Estrutura

```
pjbl2-chatbot-cs2/
├── chat.py                     atalho: python chat.py --versao v1
├── src/
│   ├── lexer_afd.py            lexer por AFD (adaptado do analisador_lexico.c da Aula 02)
│   ├── normalizacao.py         v0
│   ├── base.py                 base de conhecimento, grupos de resposta, siglas
│   ├── similaridade.py         cosseno em NumPy
│   ├── assistente.py           interface comum (Resultado, ranking, limiar)
│   ├── chatbot_tfidf.py        v1
│   ├── chatbot_denso.py        v2
│   ├── avaliacao.py            métricas, limiares, experimentos, ablação, relatórios
│   ├── chat.py                 interface de linha de comando
│   ├── preparar_recursos.py    baixa dados do NLTK e modelos
│   ├── teste_informal.py       mesma pergunta formal, sem acento e abreviada
│   └── bonus_lora_dora.py      bônus
├── tests/                      testes automatizados (unittest)
├── dados/
│   ├── base.csv                id, pergunta, resposta, fonte, categoria (100 pares)
│   ├── validacao.csv           32 perguntas (24 do escopo + 8 fora) – ajuste de limiar/decisões
│   └── teste.csv               42 perguntas (34 do escopo + 8 fora) – só medição final
├── resultados/                 gerado pelo src/avaliacao.py (comparacao.csv etc.)
├── docs/
│   ├── decisoes.md             decisões + análise de erros
│   └── resultados_experimentos.md   gerado pelo src/avaliacao.py
├── requirements.txt
└── README.md
```

## Como executar (ambiente limpo, CPU)

Requer **Python 3.10+** e internet na primeira execução (dados do NLTK e modelos do
Hugging Face).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# (opcional, deixa o download do torch bem menor – só CPU):
# pip install torch --index-url https://download.pytorch.org/whl/cpu

python src/preparar_recursos.py      # NLTK (punkt_tab, stopwords, rslp) + modelo da v2 (~470 MB)
python -m unittest discover -s tests -v
python src/avaliacao.py              # gera resultados/ e preenche as tabelas do docs/decisoes.md
python src/teste_informal.py         # verificação extra de linguagem informal (seção 8.1)
```

A avaliação completa compara também o modelo alternativo
(`paraphrase-multilingual-mpnet-base-v2`, ~1 GB de download). Para pular:
`python src/avaliacao.py --sem-alternativo`. Para rodar só a v1 (sem torch):
`python src/avaliacao.py --sem-denso`.

### Conversar com o assistente

```bash
python chat.py --versao v1
python chat.py --versao v2
python chat.py --versao ambas                                 # v1 e v2 respondendo a mesma pergunta
python chat.py --versao v1 --ranking 3                        # mostra os 3 primeiros do ranking
python chat.py --versao v2 --pergunta "quanto custa o prime?"  # uma pergunta só
```

Formato da saída (escores e limiar ilustrativos — os reais dependem da execução):

```
> qnto custa o prime
[v1] (id=41, escore=0.47) É uma compra única, sem assinatura. Na loja americana custa US$ 14,99; ...
> qual a senha do wifi?
[v1] Desculpe, não entendi. Pode reformular? (escore=0.00 < limiar 0.30)
```

Os limiares usados pelo chat vêm de `resultados/limiares.json`, escrito pela avaliação
(antes disso valem os padrões 0,30 para a v1 e 0,60 para a v2). Entradas vazias, só
pontuação, emojis, textos enormes ou em outro idioma geram resposta controlada.

### Bônus (LoRA / DoRA)

```bash
python src/bonus_lora_dora.py                                  # ~2 a 5 min em CPU
python src/bonus_lora_dora.py --pergunta "como denuncio um hacker?"
```

Treina um classificador de categoria da pergunta sobre o encoder da v2 de três formas
(só a camada final, LoRA e DoRA com `use_dora=True`) e mostra: a perda caindo a cada época,
a tabela de **parâmetros treináveis × total** e a categoria prevista para perguntas novas.
Salva `resultados/bonus_lora_dora.csv` e `resultados/bonus_exemplos.csv` e preenche a seção 9
do `docs/decisoes.md`, onde o mecanismo é explicado.

## Reprodutibilidade

- Semente fixa (42) em `random`, NumPy e torch.
- `resultados/ambiente.json` registra versões do Python, NLTK, scikit-learn, NumPy,
  pandas, sentence-transformers, torch e o commit do modelo no Hugging Face.
- Os embeddings da base ficam em `cache/` e são recalculados só se a base ou o modelo mudarem.

## Uso do código do professor

`src/lexer_afd.py` é a tradução para Python do `analisador_lexico.c` (Aula 02): o mesmo
AFD de números (estados, função δ e estratégia de maior prefixo), mais um AFD de
palavras. Ele classifica tokens (para proteger siglas como VAC/VRS/MR12) e é um
tokenizador alternativo comparado ao NLTK na ablação. Detalhes em `docs/decisoes.md`, seção 2.2.

## Bibliotecas

Somente as permitidas no enunciado: NLTK, scikit-learn, NumPy, pandas,
sentence-transformers (com torch), transformers e PEFT (bônus). Testes com `unittest`
da biblioteca padrão. Nenhuma biblioteca extra.

## Fontes da base de conhecimento

Respostas adaptadas e traduzidas para o português a partir de (consultadas em nov/2026):

**Oficiais (Valve / Steam)**
- Steam Support – CS2 Release: https://help.steampowered.com/pt-br/faqs/view/5ED2-ED8E-81F4-0C18
- Steam Support – CS2 Prime Status e Free to Play: https://help.steampowered.com/pt-br/faqs/view/4D81-BB44-4F5C-9B6B
- Steam Support – CS2 Patentes e nível de perfil: https://help.steampowered.com/pt-br/faqs/view/3204-9FB1-967F-1AC6
- Steam Support – CS2 Leaderboards: https://help.steampowered.com/pt-br/faqs/view/1BCF-10D0-1AF1-302C
- Steam Support – CS2 Cooldowns e banimentos competitivos: https://help.steampowered.com/pt-br/faqs/view/7580-7BDB-1C9A-AC67
- Steam Support – Trust Factor: https://help.steampowered.com/pt-br/faqs/view/00EF-D679-C76A-C185
- Steam Support – Fui banido pelo VAC: https://help.steampowered.com/pt-br/faqs/view/647C-5CC1-7EA9-3C29
- Steam Support – Denunciar trapaceiros: https://help.steampowered.com/pt-br/faqs/view/2F3F-25EE-2AC6-30E7
- Steam Support – Arsenal (Armory): https://help.steampowered.com/pt-br/faqs/view/2DBD-7CEA-8206-2B00
- Steam Support – Retenções de troca e Mercado: https://help.steampowered.com/pt-br/faqs/view/34A1-EA3F-83ED-54AB
- Steam Support – Restrições de troca/Mercado: https://help.steampowered.com/pt-br/faqs/view/451E-96B3-D194-50FC
- Steam Support – Verificar integridade dos arquivos: https://help.steampowered.com/pt-br/faqs/view/0C48-FCBD-DA71-93EB
- Steam Support – Requisitos do CS2: https://help.steampowered.com/pt-br/wizard/HelpWithGameTechnicalIssue?appid=730
- Política de reembolso da Steam: https://store.steampowered.com/steam_refunds/
- Valve Developer Community – Counter-Strike 2: https://developer.valvesoftware.com/wiki/Counter-Strike_2
- Valve – Regional Standings (GitHub): https://github.com/ValveSoftware/counter-strike_regional_standings
- Fórum oficial da Steam (resposta do suporte sobre o Prime gratuito do CS:GO): https://steamcommunity.com/app/730/discussions/0/4417550166392615402

**Especializadas / organizadores**
- PGL – Major Singapore 2026: https://www.pglesports.com/cs2/pgl-major-singapore-2026/
- swap.gg – Majors de 2026: https://swap.gg/blog/cs2-majors-2026
- bo3.gg – Bomb Defusal: https://bo3.gg/wiki/bomb-defusal
- csdb.gg – rodadas (MR12), Wingman, desarme, surrender, caixas, Proteção de Troca:
  https://csdb.gg/guides/how-many-rounds-cs2/ · https://csdb.gg/guides/what-is-wingman-cs2/ ·
  https://csdb.gg/guides/how-to-defuse-cs2/ · https://csdb.gg/how-to-surrender-in-cs/ ·
  https://csdb.gg/guides/how-to-get-cs2-cases/ · https://csdb.gg/cs2-trade-protection-update/
- HotSpawn – economia do CS2: https://www.hotspawn.com/counter-strike/guide/how-to-read-the-economy-in-cs2
- Refrag – loss bonus: https://refrag.gg/blog/cs2-economy-crash-course-what-are-kill-rewards-and-loss-bonus
- GuildOrder – recompensas por abate: https://guildorder.com/games/csgo/wiki/economy-basics
- cs2lab – limite de dinheiro: https://cs2lab.pro/guide-to-understand-the-csgo-economy/
- Alviran – pool Active Duty 2026: https://alviran.net/blog/cs2-map-pool-2026-active-duty-history/
- Metabot – cores e temporadas do CS Rating: https://metabot.gg/en/cs2/guides/cs2-premier-rating-explained
- Counter-Strike Wiki (Fandom) – Premier e Overwatch: https://counterstrike.fandom.com/wiki/Premier · https://counterstrike.fandom.com/wiki/Overwatch
- iTechGuides – CS Rating × Competitivo: https://www.itechguides.com/how-cs2-matchmaking-ranks-and-premier-rating-work/
- BitTopup – Premier × Competitivo (prorrogação): https://bittopup.com/article/cs2-premier-vs-competitive-2026
- Dot Esports – pick/ban do Premier: https://dotesports.com/counter-strike/news/counter-strike-2-premier-cs2-rating-explained
- SenseOS – cooldown do VACNet: https://senseos.gg/guides/counter-strike-2/competitive-cooldowns/
- Skinport – reverter trocas: https://skinport.com/blog/how-to-reverse-cs2-trades
- Pulse (Dot Esports) – preço do Prime: https://www.pulse.bot/gaming/news/cs2-prime-matchmaking-weekly-drops-and-is-it-worth-the-price-fa624a7c-5eb7-46c4-b60e-865a583ed509/

Nenhum dado pessoal de terceiros foi usado.
