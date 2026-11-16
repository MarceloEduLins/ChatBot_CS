# Decisões de projeto e análise de erros

**PjBL 2 – Chatbot de FAQ (Construção de Interpretadores, PUCPR, 2026/2)**
Dupla: Marcelo Lins e Miguel Wihby — Domínio: **Counter-Strike 2 (competitivo e suporte ao jogador)**

> As tabelas marcadas com `AUTO` são escritas pelo `python src/avaliacao.py`
> (não editar à mão, elas são regeradas a cada execução). Os números completos
> ficam em `resultados/` e em `docs/resultados_experimentos.md`.
> As leituras e a análise de erros foram escritas com os números da execução de
> 15/11/2026 (`resultados/ambiente.json`).

---

## 1. Domínio e base de conhecimento

- **Domínio:** perguntas frequentes de jogadores de CS2 sobre o modo competitivo
  (Premier, CS Rating, patentes, regras de rodada, economia, Majors) e sobre
  suporte (Prime, cooldowns, VAC, Trust Factor, trocas, reembolso, problemas técnicos).
- **Tamanho:** 100 pares em `dados/base.csv` (`id, pergunta, resposta, fonte, categoria`),
  com 95 respostas distintas (ver decisão 3.4 sobre duplicadas).
- **Fontes:** sempre que existia, usamos a fonte oficial da Valve (Steam Support,
  loja Steam, Valve Developer Community, GitHub da Valve). Para regras de jogo que a
  Valve não documenta em FAQ (economia, tempos de rodada, Wingman, pool de mapas)
  usamos sites especializados e com histórico no cenário (bo3.gg, csdb.gg, HotSpawn,
  Refrag, PGL). Cada linha da base tem a URL da fonte; a lista completa está no README.
  As respostas foram **adaptadas/traduzidas** por nós para o português, sem copiar
  parágrafos.
- **Coluna extra `categoria`** (11 categorias): não é usada pela v1/v2; serve para
  o bônus (classificação com LoRA/DoRA) e para organizar a base.
- **Data das informações:** parte do conteúdo muda com atualizações do jogo (pool de
  mapas, Majors de 2026). As respostas refletem o estado em **novembro de 2026** e
  precisariam de manutenção — limitação comum de bots de FAQ por recuperação.

### Conjuntos de validação e teste

| Conjunto | Escopo (id da base) | Fora do escopo (id = -1) | Uso |
|---|---|---|---|
| `dados/validacao.csv` | 24 | 8 | escolher limiar, campo indexado, normalizar ou não (v2), modelo |
| `dados/teste.csv` | 34 | 8 | **só** medir no final |

- Cada `id_esperado` foi conferido um a um contra a pergunta e a resposta da base antes da
  medição final (um rótulo errado faria o assistente "errar" uma pergunta que ele acertou).
- Nenhuma pergunta de validação ou teste aparece literalmente na base nem nos dois
  conjuntos ao mesmo tempo (`checar_contaminacao` em `src/base.py` e `tests/test_dados.py`).
- O teste varia **registro** ("Que jogo é o Counter Strike 2…" × "qnto custa o prime em dolar"),
  **gírias e sinônimos** (grana, quita, estorno, crash, season, defuse), **erros de digitação
  e falta de acento** ("da pra", "nivel", "qnto", "oq"), **siglas** (VAC, CT, AWP, CS2),
  **negação** ("não consigo entrar no premier") e **ordem das palavras**.
- Fora do escopo: além de perguntas sem relação nenhuma (wifi, Copa de 2002), incluímos
  perguntas **perto do domínio** (Fortnite, Valorant, League of Legends, RTX 4060) e
  perguntas em **inglês**, que são as mais difíceis de rejeitar.

**Possíveis vieses (discutidos):**
1. As perguntas de teste foram escritas pela própria dupla, que conhece a base. Mesmo
   tentando variar o vocabulário, a tendência é reaproveitar palavras da base, o que
   favorece o TF-IDF. Mitigação: evitamos repetir o verbo principal da pergunta
   original e usamos gírias/erros de digitação reais. Não houve revisão do teste por alguém que
   não montou a base, então esse viés continua e é a principal limitação da medição. O que
   fizemos foi conferir cada `id_esperado` um a um contra a base (ver acima). Numa próxima
   versão, cada integrante reescreveria as perguntas da parte da base que não montou, ou
   pediríamos perguntas a colegas de outro grupo.
2. O teste é pequeno: com 34 perguntas do escopo, **cada pergunta vale ~3 pontos
   percentuais** de acurácia, e com 8 perguntas fora do escopo cada uma vale 12,5 p.p.
   de rejeição. Diferenças pequenas entre v1 e v2 não são conclusivas.
3. As gírias são as que nós usamos (público de CS brasileiro); outro público escreveria
   diferente.

---

## 2. v0 – Pipeline de normalização (`src/normalizacao.py`)

Função pedida: `normalizar(texto, **etapas) -> list[str]`. Todas as etapas podem ser
ligadas/desligadas por parâmetro (`ConfigNormalizacao`).

| # | Etapa | Adotada? | Justificativa |
|---|---|---|---|
| 1 | Tokenização (NLTK `word_tokenize`) | Sim | Separa pontuação e mantém tokens como `CS2`, `2v2`. Alternativa: **lexer AFD da Aula 02** (ver 2.2), comparado na ablação. |
| 2 | Minúsculas | Sim | "Premier" e "premier" precisam ser o mesmo termo no TF-IDF. Siglas também vão para minúsculas, mas são protegidas nas etapas 4–5. |
| 3 | Remover pontuação e símbolos | Sim | `?`, `$`, emojis não carregam o assunto da pergunta. Emoji vira token INVALIDO e é descartado (o programa não trava). |
| 4 | Remover stopwords (lista do NLTK **ajustada**) | Sim | Palavras como "o", "de", "que" aparecem em quase todas as perguntas e só geram ruído. **Ajuste:** tiramos da lista as negações (`não, nem, nunca, sem, jamais, nenhum`). Interrogativos ("quanto", "quando", "como") ficam como opção e foram medidos. |
| 5 | Stemming (RSLP do NLTK) | Sim | Junta flexões como "patente/patentes", "vitória/vitórias", "custa/custo". Reduz a esparsidade, que é o maior problema do TF-IDF com só 100 documentos. Não resolve sinônimos ("grana" × "dinheiro"). |
| 6 | Remover acentos | Sim, **depois** do stemming | O usuário escreve "nivel", "maximo", "nao". O RSLP usa os sufixos acentuados (-ção, -ões), então tirar acento antes estragaria o stemming. |
| – | Lematização (spaCy) | **Não** | O enunciado pede stemming **ou** lematização. O `pt_core_news_sm` acrescenta uma dependência pesada, é mais lento e erra bastante em texto informal ("tá", "qnto", "quita"). Para o TF-IDF o que importa é fundir variações, e o RSLP faz isso. Fica como trabalho futuro. |
| extra | Proteger siglas | Sim | Ver 2.1. |
| extra | Expandir abreviações de chat (etapa 0) | Sim | Ver 2.1. |
| extra | Marcar negação (`nao_`) | Opcional, medida | Ver 2.1. |

### Quadro antes/depois (5 perguntas da base + 1 exemplo de negação)

<!-- AUTO:quadro_v0 -->
| origem | antes (texto original) | só tokenização | pipeline ingênuo (stopwords NLTK puras, sem siglas, sem abreviações) | pipeline adotado |
|---|---|---|---|---|
| id 73 | Fui banido pelo VAC sem usar trapaça, o que faço? | Fui · banido · pelo · VAC · sem · usar · trapaça · , · o · que · faço · ? | ban · vac · us · trap · fac | ban · vac · sem · us · trap · fac |
| id 100 | O que é o VRS (Valve Regional Standings)? | O · que · é · o · VRS · ( · Valve · Regional · Standings · ) · ? | vrs · valv · region · standing | vrs · valv · region · standing |
| id 16 | Vitórias no modo Competitivo aumentam meu CS Rating? | Vitórias · no · modo · Competitivo · aumentam · meu · CS · Rating · ? | vitor · mod · competi · aument · cs · rating | vitor · mod · competi · aument · cs · rating |
| id 43 | Eu tinha Prime grátis do CS:GO e ele sumiu, o que aconteceu? | Eu · tinha · Prime · grátis · do · CS · : · GO · e · ele · sumiu · , · o · que · aconteceu · ? | prim · gratil · cs · go · sum · acontec | prim · gratil · cs · go · sum · acontec |
| id 85 | O CS2 roda no Mac? | O · CS2 · roda · no · Mac · ? | cs2 · rod · mac | cs2 · rod · mac |
| extra (fora da base) | não consigo entrar no premier, o botão tá cinza | não · consigo · entrar · no · premier · , · o · botão · tá · cinza | consig · entr · premi · bot · ta · cinz | nao · consig · entr · premi · bot · cinz |
| extra (fora da base) | qnto custa o prime? pq n consigo comprar | qnto · custa · o · prime · ? · pq · n · consigo · comprar | qnt · cust · prim · pq · n · consig · compr | quant · cust · prim · nao · consig · compr |
<!-- /AUTO:quadro_v0 -->

### 2.1 Onde a normalização destrói informação (e o que fizemos)

**Negação.** A lista padrão do NLTK contém "não", "nem" e "sem". Com ela, as perguntas
"Recebo patente e CS Rating **sem** ter o Prime?" (id 38) e uma hipotética "Recebo
patente e CS Rating **tendo** o Prime?" ficam praticamente iguais, e "não consigo entrar no premier" perde justamente a
palavra que diz que é um problema. Solução adotada: **manter as negações** (tirar da lista
de stopwords). Também implementamos uma segunda forma, opcional: **marcar a negação**
— a primeira palavra de conteúdo depois de "não/nem/nunca/sem" ganha o prefixo `nao_`
("não consigo" → `nao · nao_consig`), com o escopo terminando na pontuação. A ideia
vem da análise sintática: em vez de olhar só a palavra solta, olhamos a estrutura
`NEG → negação PALAVRA`. Na base quase todas as perguntas são afirmativas, então marcar
a negação tende a **separar** a pergunta do usuário da pergunta da base (o termo
`nao_consig` não existe no vocabulário). Por isso ela ficou desligada por padrão — a
ablação mede isso.

**Siglas.** O RSLP trata siglas como palavras: a regra de plural poderia cortar o "s"
final de `VRS`, e `CS` poderia sumir; e o usuário digita a mesma sigla em minúsculas
("vac", "cs2"). Solução: o **lexer AFD** (2.2) classifica como `SIGLA` todo token só
com maiúsculas/dígitos (VAC, MR12, CS2, CTs), e montamos um **dicionário de siglas
extraído da própria base** (`Base.siglas`: vac, vrs, cs2, mr12, ct, tr, awp, afk, xp…).
Um token que é sigla pela caixa **ou** está no dicionário não passa por stopwords nem
por stemming. Plural de sigla é unificado (`CTs` → `ct`). Se o texto inteiro está em
CAPS LOCK, a caixa deixa de valer como pista e só o dicionário conta.

**O que a medição mostrou:** na ablação (5.1), desligar a proteção não mudou nenhuma
métrica nem o vocabulário (638 termos). Nesta base o RSLP não chegou a estragar as
siglas: no quadro acima, VRS, VAC e CS2 saem intactos até no pipeline ingênuo.
Mantivemos a proteção como salvaguarda, porque não custa nada e cobre siglas novas que
entrarem na base.

**Stemming agressivo.** O RSLP às vezes junta palavras de sentidos diferentes e às vezes
separa formas que deveriam se juntar. Casos que apareceram de verdade nos resultados:

- "Como calcular o determinante de uma matriz 3x3?" (fora do escopo, teste) foi **aceita**
  pela v1 e respondida com o id 35, porque "calcular" e "calculadas" (da resposta do id 35)
  viram o mesmo radical `calcul`;
- "bans" virou `bam` (a regra de plural `-ns → -m` do RSLP, a mesma de "homens → homem"),
  que não casa com o `ban` de "banido";
- "grátis" virou `gratil` (quadro acima).

Mesmo com esses erros, o stemming é a etapa que mais ajuda no saldo (ablação 5.1). A
lematização evitaria parte deles, ao custo citado na tabela do início desta seção.

**Acentos.** Remover acentos junta "é" (verbo) e "e" (conjunção), mas os dois já são
stopwords. Em troca, "nivel"/"nível" e "maximo"/"máximo" passam a casar — ganho maior.

**Abreviações de chat.** O enunciado pede que o teste varie o registro (formal/informal),
e quem joga CS escreve "vc", "pq", "qnto", "n". Sem tratamento, essas formas viram tokens que
não existem no vocabulário da base: o TF-IDF simplesmente as ignora, e quando a palavra
abreviada carregava o sentido ("qntas" × "quantas") a pergunta perde o termo que casaria. Pior:
"n" no lugar de "não" apaga a negação. Solução: um **dicionário de abreviações**
(`ABREVIACOES` em `src/normalizacao.py`, ~45 formas) aplicado no texto, antes da tokenização
(etapa 0). Só troca palavras soltas ("praia", "3d" e "pick-ban" não mudam) e deixa de fora
formas ambíguas no nosso domínio ("to", "num"). A segunda linha extra do quadro acima mostra o
efeito: "qnto custa o prime? pq n consigo comprar" vira `quant · cust · prim · nao · consig ·
compr`, contra `qnt · cust · prim · pq · n · consig · compr` no pipeline ingênuo. A v2 usa o
mesmo dicionário na pergunta do usuário (seção 4). O efeito foi medido na validação (4.1 e
5.1) e num teste específico de linguagem informal (8.1).

### 2.2 Reaproveitamento do código do professor (Aula 02 – `analisador_lexico.c`)

O `src/lexer_afd.py` é uma tradução do `analisador_lexico.c` para Python:

- **mesmo AFD de números** (`Q_INICIO … Q_EXP_DIGITO`, mesma função δ, mesma estratégia
  de *maior prefixo* do `reconhece_numero`), com uma adaptação: a vírgula também é
  separador decimal (`14,99`, `16.000`). O teste `test_mesmos_casos_do_codigo_em_c`
  confere os mesmos 8 casos do `testar_afd()` do C;
- um **segundo AFD para palavras** (letras, dígitos e hífen interno: `pick-ban`, `2v2`),
  no lugar dos operadores da calculadora;
- o lexer usa a mesma ideia do `espera_numero` do C e escolhe o maior prefixo entre os
  dois AFDs (`2v2` vira palavra, `16.000` vira número);
- diferença proposital: no C o lexer **para** no primeiro símbolo inválido; aqui o
  símbolo vira `INVALIDO` e a análise continua, porque o chatbot não pode travar com
  um emoji.

- o teste `test_afd_equivale_a_regex` mostra na prática o **MEF ≡ Regex** da Aula 02: em
  3000 strings aleatórias, o AFD de números aceita exatamente as mesmas strings (e o mesmo
  maior prefixo) que a expressão regular `[+-]?[0-9]+([.,][0-9]+)*([eE][+-]?[0-9]+)?`.

Uso no projeto: (1) **classificar** cada token (SIGLA, NUMERO, PALAVRA, PONTUACAO),
inclusive os que vêm do NLTK — é isso que permite proteger siglas; (2) **tokenizador
alternativo** ao NLTK, comparado na ablação. A marcação de negação (`NEG → negação
PALAVRA`) segue a mesma lógica de gramática usada no `analisador_sintatico.c`, em
versão bem simplificada.

---

## 3. v1 – TF-IDF + cosseno (`src/chatbot_tfidf.py`)

- **Vetorização:** `TfidfVectorizer(analyzer=normalizador)`, ou seja, o próprio pipeline
  v0 gera os termos. Fórmula do scikit-learn (padrão `smooth_idf=True`):
  `idf(t) = ln((1+N)/(1+df(t))) + 1`, `tfidf = tf · idf`, depois normalização L2.
- **Cosseno implementado pela dupla** em `src/similaridade.py`:
  `cos(q,d) = (q·d) / (‖q‖·‖d‖)`, com proteção para vetor nulo (pergunta só com
  stopwords → vetor zero → similaridade 0, em vez de divisão por zero). Conferido contra
  `sklearn.metrics.pairwise.cosine_similarity` em `tests/test_similaridade.py`
  (vetores aleatórios, matriz TF-IDF real da base, vetor nulo e valores conhecidos).
- **Interface:** `responder(texto)` devolve `Resultado(resposta, id, escore, ranking)`
  — a mesma da v2 (`src/assistente.py`), então a avaliação trata as duas igual.

### 3.1 O que indexar (decisão medida na validação)

Testamos indexar só a pergunta, só a resposta, ou pergunta + resposta. Indexar a
resposta aumenta o vocabulário (a pergunta do usuário às vezes usa palavras que só
aparecem na resposta, como "estorno"/"reembolso" ou "crash"/"crashando"), mas também
traz ruído (respostas longas compartilham termos como "partida", "Steam"). Critério:
maior MRR na validação (desempate pela acurácia@1).

<!-- AUTO:indexacao -->
| versao | campo | limiar_val | val_acuracia@1 | val_mrr | teste_acuracia@1 | teste_mrr |
|---|---|---|---|---|---|---|
| v1 | pergunta | 0.710 | 0.292 | 0.418 | 0.412 | 0.570 |
| v1 | resposta | 0.660 | 0.292 | 0.417 | 0.471 | 0.558 |
| v1 | pergunta+resposta | 0.330 | 0.417 | 0.530 | 0.588 | 0.716 |

Campo escolhido: **pergunta+resposta**.
<!-- /AUTO:indexacao -->

### 3.2 Limiar de rejeição (sem olhar o teste)

- O limiar é escolhido **só na validação**, varrendo de 0,00 a 1,00 (passo 0,01).
- Objetivo: `J = 0,5·(perguntas do escopo respondidas com o id certo) + 0,5·(perguntas
  fora do escopo rejeitadas)`. O peso igual evita que as 24 perguntas do escopo dominem
  as 8 de fora (se otimizássemos a acurácia geral, o limiar tenderia a zero e o bot
  nunca diria "não entendi").
- Empate: pegamos a **maior faixa contínua** de limiares empatados e o limiar do
  **meio** dela, que é o menos sensível a pequenas variações.
- v1 e v2 têm limiares **diferentes**: o cosseno do TF-IDF costuma ficar perto de 0
  quando não há palavras em comum, enquanto embeddings dão 0,2–0,4 até para frases sem
  relação. Os valores escolhidos ficam em `resultados/limiares.json`, que o `chat.py` lê.

<!-- AUTO:limiares -->
```json
{
  "v1": {
    "limiar": 0.33,
    "campo": "pergunta+resposta",
    "expandir_abreviacoes": true
  },
  "v2": {
    "limiar": 0.54,
    "modelo": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "campo": "pergunta+resposta",
    "normalizar": false,
    "expandir_abreviacoes": true
  }
}
```
<!-- /AUTO:limiares -->

Curva do limiar da v1 na validação (troca entre rejeitar fora do escopo e rejeitar
indevidamente perguntas válidas):

<!-- AUTO:curva_v1 -->
| limiar | escopo_respondido_certo | rejeicao_correta | falsa_rejeicao | J |
|---|---|---|---|---|
| 0.000 | 0.417 | 0.000 | 0.000 | 0.208 |
| 0.050 | 0.417 | 0.125 | 0.042 | 0.271 |
| 0.100 | 0.417 | 0.125 | 0.042 | 0.271 |
| 0.150 | 0.417 | 0.250 | 0.042 | 0.333 |
| 0.200 | 0.417 | 0.250 | 0.042 | 0.333 |
| 0.250 | 0.375 | 0.500 | 0.167 | 0.438 |
| 0.300 | 0.375 | 0.750 | 0.250 | 0.562 |
| 0.350 | 0.375 | 0.875 | 0.417 | 0.625 |
| 0.400 | 0.167 | 0.875 | 0.667 | 0.521 |
| 0.450 | 0.125 | 0.875 | 0.833 | 0.500 |
| 0.500 | 0.042 | 0.875 | 0.958 | 0.458 |
| 0.550 | 0.042 | 1.000 | 0.958 | 0.521 |
| 0.600 | 0.042 | 1.000 | 0.958 | 0.521 |
| 0.650 | 0.042 | 1.000 | 0.958 | 0.521 |
| 0.700 | 0.042 | 1.000 | 0.958 | 0.521 |
| 0.750 | 0.042 | 1.000 | 0.958 | 0.521 |
| 0.800 | 0.042 | 1.000 | 0.958 | 0.521 |
| 0.850 | 0.000 | 1.000 | 1.000 | 0.500 |
| 0.900 | 0.000 | 1.000 | 1.000 | 0.500 |
| 0.950 | 0.000 | 1.000 | 1.000 | 0.500 |
| 1.000 | 0.000 | 1.000 | 1.000 | 0.500 |
<!-- /AUTO:curva_v1 -->

### 3.3 Empates de escore

O ranking é ordenado por escore decrescente e, em empate exato, pelo **menor id**.
É uma regra determinística (o mesmo resultado em toda execução) e simples de explicar.
Empates exatos acontecem quase só quando o escore é 0 (nenhum termo em comum) — e aí a
pergunta é rejeitada pelo limiar de qualquer jeito.

### 3.4 Perguntas da base com a mesma resposta

A base tem 5 pares que são reformulações com resposta idêntica (11/12, 64/65, 72/73,
87/88, 92/93): no FAQ real a mesma dúvida aparece escrita de jeitos diferentes, e ter as
duas formulações ajuda o TF-IDF. Para não distorcer as métricas:

- cada resposta tem um **id canônico** (o menor id do grupo) — `Base.grupo_de`;
- o ranking é **por resposta**: o grupo entra uma vez só, com o **maior** escore entre
  suas perguntas (max-pooling). Sem isso, a mesma resposta ocuparia a 1ª e a 2ª posição
  e empurraria a resposta certa para baixo, piorando o MRR injustamente;
- o `id_esperado` do teste é convertido para o id canônico antes de comparar.

---

## 4. v2 – Embeddings densos (`src/chatbot_denso.py`)

- **Modelo padrão:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (sugerido no enunciado): ~118M parâmetros, vetores de 384 dimensões, roda em CPU e
  foi treinado justamente para **paráfrases** em 50+ línguas — que é a nossa tarefa
  (pergunta do usuário × pergunta da base).
- **Alternativa comparada:** `paraphrase-multilingual-mpnet-base-v2` (mesma família,
  maior: 768 dimensões). A escolha final é a de maior MRR na validação; se empatar,
  fica o menor (mais rápido).
- **Cache:** os embeddings da base são calculados uma vez e salvos em
  `cache/emb_<modelo>_<cru|norm>_<hash>.npy`. O hash depende do modelo, dos textos e de
  normalizar ou não, então se a base mudar o cache é refeito sozinho.
- **Abreviações:** a pergunta do usuário passa pelo mesmo dicionário de abreviações da v1
  (2.1) antes do embedding. A base não tem abreviações, então o texto dela e o cache não mudam.
- **Mesma interface e mesma lógica de ranking/limiar da v1** (classe base comum), com
  **limiar próprio** escolhido na validação.
- **Reprodutibilidade:** semente 42 (random, NumPy, torch); a versão do modelo
  (commit do Hugging Face), do `sentence-transformers` e do `torch` vão para
  `resultados/ambiente.json`.

### 4.1 Normalizar antes dos embeddings? (decisão medida)

Hipótese inicial: **não normalizar**. O Transformer foi treinado com texto natural; ele
usa as stopwords e a ordem das palavras para montar o contexto (é isso que a atenção
faz), e os radicais do RSLP ("consig", "entr") nem são palavras que o tokenizador do
modelo conhece bem. Medimos as duas opções na validação:

<!-- AUTO:experimentos_v2 -->
| etapa | modelo | campo | normalizar_antes | expandir_abreviacoes | limiar_val | val_acuracia@1 | val_mrr | teste_acuracia@1 | teste_mrr |
|---|---|---|---|---|---|---|---|---|---|
| campo | paraphrase-multilingual-MiniLM-L12-v2 | pergunta | False | True | 0.540 | 0.500 | 0.600 | 0.676 | 0.759 |
| campo | paraphrase-multilingual-MiniLM-L12-v2 | resposta | False | True | 0.540 | 0.292 | 0.440 | 0.471 | 0.588 |
| campo | paraphrase-multilingual-MiniLM-L12-v2 | pergunta+resposta | False | True | 0.480 | 0.500 | 0.638 | 0.706 | 0.777 |
| normalizacao | paraphrase-multilingual-MiniLM-L12-v2 | pergunta+resposta | True | True | 0.540 | 0.292 | 0.433 | 0.412 | 0.515 |
| modelo | paraphrase-multilingual-mpnet-base-v2 | pergunta+resposta | False | True | 0.540 | 0.625 | 0.738 | 0.735 | 0.820 |
| abreviacoes | paraphrase-multilingual-mpnet-base-v2 | pergunta+resposta | False | False | 0.540 | 0.625 | 0.737 | 0.735 | 0.827 |
<!-- /AUTO:experimentos_v2 -->

**Leitura.** Normalizar antes atrapalhou muito a v2: com o MiniLM, a acurácia@1 na
validação caiu de 0,500 para 0,292 e o MRR de 0,638 para 0,433. Isso confirma a hipótese:
os radicais do RSLP não são palavras que o modelo viu no treino e, sem as stopwords, a frase
perde o contexto que a atenção usa. Indexar pergunta + resposta ficou acima de só a pergunta
(MRR 0,638 × 0,600) e bem acima de só a resposta (0,440). O modelo maior (mpnet) melhorou a
validação (acurácia@1 0,625 × 0,500; MRR 0,738 × 0,638) e por isso foi o escolhido, pelo
critério definido antes de medir (maior MRR na validação). O custo é real, mas aceitável com
100 perguntas: cerca de 278 milhões de parâmetros contra 118 milhões, vetores de 768 dimensões
em vez de 384 e um download de ~1,1 GB em vez de ~470 MB.

**Abreviações.** A última linha desliga o dicionário de abreviações no modelo escolhido. A
regra, definida antes de medir, foi manter o dicionário (ele vem do pedido de variar o registro)
a não ser que desligá-lo fosse melhor na validação. Não foi: o MRR ficou em 0,738 com o
dicionário e 0,737 sem, com a mesma acurácia@1. A diferença é mínima porque a validação quase
não tem abreviações (só "pra", cinco vezes). O ganho real aparece no teste de linguagem
informal (8.1).

---

## 5. Comparação experimental (conjunto de teste)

<!-- AUTO:comparacao -->
| versao | representacao | campo | limiar | acuracia@1 | mrr | rejeicao_correta | falsa_rejeicao | acerto_final |
|---|---|---|---|---|---|---|---|---|
| v1 | TF-IDF | pergunta+resposta | 0.330 | 0.588 | 0.716 | 0.750 | 0.412 | 0.500 |
| v2 | embeddings (paraphrase-multilingual-mpnet-base-v2) | pergunta+resposta | 0.540 | 0.735 | 0.820 | 1.000 | 0.324 | 0.619 |
<!-- /AUTO:comparacao -->

Como ler:
- **Acurácia@1 e MRR** medem só o ranking (não dependem do limiar). O MRR premia quem
  coloca a resposta certa em 2º ou 3º lugar — útil para ver se o erro foi "por pouco".
- **Rejeição correta × falsa rejeição** dependem do limiar e andam juntas: subir o
  limiar rejeita mais perguntas fora do escopo, mas também rejeita perguntas válidas.
  A curva da seção 3.2 mostra essa troca.
- **acerto_final** (métrica extra) junta tudo: do escopo, respondeu o id certo e não
  rejeitou; de fora, rejeitou. É o que o usuário realmente sente.

**Leitura crítica.**

- **A v2 ganhou em todas as métricas.** Acurácia@1 de 73,5% × 58,8% (25 × 20 das 34
  perguntas do escopo), MRR 0,820 × 0,716, rejeição correta de 100% × 75% (8 × 6 das 8 de
  fora), falsa rejeição de 32,4% × 41,2% (11 × 14) e acerto final de 61,9% × 50,0% (26 × 21
  das 42).
- **A diferença na acurácia@1 é de 5 perguntas**, acima do ruído de uma pergunta (~3 p.p.),
  mas o teste é pequeno. Olhando só as perguntas em que as versões discordam, a v2 acertou
  9 que a v1 errou e a v1 acertou 4 que a v2 errou. Com 13 discordâncias, o teste de McNemar
  exato dá p ≈ 0,27: a tendência favorece a v2, mas não dá para afirmar com segurança
  estatística.
- **Perguntas fora do escopo "perto do domínio".** A v1 aceitou duas: "Quando começa a nova
  temporada do Fortnite?" (escore 0,62, respondeu sobre as temporadas do Premier) e "Como
  calcular o determinante de uma matriz 3x3?" (0,34, ver 2.1). O caso do Fortnite mostra um
  limite do saco de palavras: "Fortnite" não está no vocabulário, então a palavra que mais
  diferencia a pergunta simplesmente não existe para o TF-IDF, e o resto ("quando começa a
  nova temporada") é quase igual ao id 18. A v2 rejeitou as oito; o maior escore dela fora do
  escopo foi 0,44 ("Qual o preço do dólar hoje?", perto do preço do Prime).
- **As duas rejeitam perguntas válidas demais.** Na v2, 7 das 11 falsas rejeições tinham a
  resposta certa em 1º lugar, mas com escore entre 0,44 e 0,53, abaixo do limiar de 0,54.
  A curva da validação explica: o J máximo (0,79) aparece em um único ponto, 0,54, onde uma
  pergunta fora do escopo a mais passa a ser rejeitada; entre 0,44 e 0,53 o J era 0,75. Com
  só 8 perguntas fora do escopo na validação, **uma pergunta decidiu o limiar**. Não mudamos o
  limiar olhando o teste (isso invalidaria a medição); a lição para uma próxima versão é ter
  mais perguntas fora do escopo na validação.
- **Efeito do dicionário de abreviações no teste.** Na v1 nada mudou: as abreviações do teste
  ("pra", "pro", "oq", "tá") viram stopwords depois de expandidas, e o "qnto" já era respondido
  certo. Na v2 a acurácia@1 ficou igual, o MRR caiu de 0,827 para 0,820 e uma falsa rejeição a
  mais apareceu: "Como funcionam os convites da Valve pros times irem ao Major?" passou de 0,545
  para 0,539 ("pros" virou "para os"). Ela já era respondida errado (id 99), então o acerto final
  não muda; o usuário passa a receber "não entendi" em vez de uma resposta errada.

### 5.1 Experimento de ablação do pipeline v0 (sobre a v1)

Cada linha liga/desliga **uma** etapa; o limiar é reescolhido na validação para cada
variante (comparar variantes com o mesmo limiar seria injusto, porque a escala dos
escores muda).

<!-- AUTO:ablacao -->
| variante | limiar_val | val_acuracia@1 | val_mrr | teste_acuracia@1 | teste_mrr | teste_rejeicao_correta | teste_falsa_rejeicao | vocabulario |
|---|---|---|---|---|---|---|---|---|
| pipeline adotado | 0.330 | 0.417 | 0.530 | 0.588 | 0.716 | 0.750 | 0.412 | 638 |
| sem expandir abreviações | 0.330 | 0.417 | 0.530 | 0.588 | 0.716 | 0.750 | 0.412 | 638 |
| sem remover stopwords | 0.350 | 0.417 | 0.563 | 0.588 | 0.705 | 0.875 | 0.588 | 690 |
| stopwords NLTK puras (remove 'não') | 0.330 | 0.417 | 0.537 | 0.618 | 0.727 | 0.750 | 0.412 | 635 |
| + marcar negação (nao_) | 0.330 | 0.375 | 0.507 | 0.618 | 0.722 | 0.750 | 0.412 | 662 |
| + manter interrogativos | 0.330 | 0.375 | 0.510 | 0.618 | 0.723 | 0.875 | 0.353 | 641 |
| sem stemming | 0.350 | 0.333 | 0.469 | 0.559 | 0.654 | 1.000 | 0.559 | 857 |
| sem remover acentos | 0.330 | 0.375 | 0.477 | 0.618 | 0.732 | 0.750 | 0.471 | 643 |
| sem proteger siglas | 0.330 | 0.417 | 0.530 | 0.588 | 0.716 | 0.750 | 0.412 | 638 |
| tokenizador AFD (lexer da Aula 02) | 0.330 | 0.417 | 0.530 | 0.529 | 0.687 | 0.750 | 0.412 | 640 |
<!-- /AUTO:ablacao -->

**Leitura.**

- **O stemming foi a etapa que mais pesou.** Sem ele o vocabulário cresce de 638 para 857
  termos (+34%) e a validação cai de 0,417 para 0,333 na acurácia@1 e de 0,530 para 0,469 no
  MRR. Com uma base pequena, juntar "consigo/conseguir" ou "vitória/vitórias" num termo só é
  o que mais ajuda.
- **Abreviações:** desligar o dicionário não mudou nenhum número da v1, nem na validação nem
  no teste. Na v1 uma abreviação desconhecida é só ignorada, e as que aparecem nesses conjuntos
  viram stopwords depois de expandidas. O efeito aparece no teste de linguagem informal (8.1).
- **Remover stopwords:** na validação, manter as stopwords dá a mesma acurácia@1 (0,417) e um
  MRR um pouco maior (0,563 × 0,530); é uma diferença pequena. Mantivemos a remoção, que faz
  parte do v0 pedido. No teste, a variante com stopwords rejeitou muito mais perguntas válidas
  (58,8% × 41,2%), com o limiar escolhido subindo para 0,35. A variante que mantém só os
  interrogativos ("quando", "quanto", "como") ficou pior na validação (0,375).
- **Negação:** tirar "não" da lista de stopwords praticamente não mudou nada (mesma
  acurácia, MRR 0,537 × 0,530), e marcar a negação com `nao_` piorou a validação (0,375),
  como previsto em 2.1.
- **Siglas:** proteger siglas não mudou nenhum número (ver 2.1).
- **Acentos:** sem remover acentos a validação piora (0,375 e MRR 0,477), porque o usuário
  escreve "nivel" e "maximo" sem acento.
- **Tokenizador AFD do professor:** na validação ficou idêntico ao do NLTK (0,417 e 0,530).
  No teste ficou duas perguntas abaixo (0,529 × 0,588), mas o teste não pode orientar
  escolhas.
- **Cuidado com o ruído:** várias variantes ficaram uma pergunta acima do pipeline adotado no
  teste (0,618 × 0,588). A escolha foi feita na validação; isso só mostra o tamanho do ruído
  com 34 perguntas.

---

## 6. Análise de erros (v1 × v2)

Casos extraídos automaticamente do teste. A coluna `dicas_hipotese` traz pistas geradas
pelo script (termos em comum, termos fora do vocabulário, siglas, negação, tamanho);
a explicação final de cada caso está logo abaixo da tabela.

<!-- AUTO:casos -->
**v2 acerta e v1 erra (9 casos)**

| pergunta | esperado | v1_top1 | v1_escore | v2_top1 | v2_escore | dicas_hipotese |
|---|---|---|---|---|---|---|
| Que jogo é o Counter Strike 2, quem fez e quando saiu? | 1 | 26 | 0.288 | 1 | 0.779 | fora do vocabulário TF-IDF: count, strik, fez; termos que puxaram para o id 26: saiu |
| meus bans antigos do csgo passaram pro cs2 tambem? | 5 | 48 | 0.341 | 5 | 0.696 | fora do vocabulário TF-IDF: bam, antig, csg; tem sigla: cs2; termos que puxaram para o id 48: pass |
| a smoke agora abre quando atiram nela? | 9 | 91 | 0.133 | 9 | 0.634 | fora do vocabulário TF-IDF: agor; termos que puxaram para o id 91: nel |
| não consigo entrar no premier, o botão tá cinza | 11 | 25 | 0.130 | 11 | 0.470 | tem negação; termos que puxaram para o id 25: entr |
| quando acaba a season o meu rating zera? | 18 | 14 | 0.183 | 18 | 0.530 | fora do vocabulário TF-IDF: season |
| Preciso de quantas vitórias pra pegar a medalha da season? | 19 | 14 | 0.420 | 19 | 0.667 | fora do vocabulário TF-IDF: peg, season; termos que puxaram para o id 14: precis, quant |
| como eu faço pra desistir da partida quando um amigo quita | 31 | 81 | 0.246 | 31 | 0.518 | fora do vocabulário TF-IDF: desist, quit |
| Minha patente da mirage desapareceu do nada | 34 | 35 | 0.235 | 34 | 0.659 | fora do vocabulário TF-IDF: desaparec, nad |
| minha troca ficou em espera de 15 dias por quê | 92 | 95 | 0.450 | 92 | 0.473 | fora do vocabulário TF-IDF: esper, que |

**v1 acerta e v2 erra (4 casos)**

| pergunta | esperado | v1_top1 | v1_escore | v2_top1 | v2_escore | dicas_hipotese |
|---|---|---|---|---|---|---|
| Onde eu compro o prime? | 40 | 40 | 0.381 | 41 | 0.561 | - |
| Qual a grana maxima que da pra juntar? | 50 | 50 | 0.314 | 56 | 0.496 | - |
| Comprei uma chave de caixa por engano, consigo estorno? | 90 | 90 | 0.304 | 97 | 0.456 | fora do vocabulário TF-IDF: estorn |
| Como funcionam os convites da Valve pros times irem ao Major? | 100 | 100 | 0.364 | 99 | 0.539 | fora do vocabulário TF-IDF: ir |

**As duas erram (5 casos)**

| pergunta | esperado | v1_top1 | v1_escore | v2_top1 | v2_escore | dicas_hipotese |
|---|---|---|---|---|---|---|
| Ganhar no competitivo sobe meu rating do premier? | 16 | 15 | 0.332 | 14 | 0.563 | termos que puxaram para o id 15: sob |
| jogador free consegue ter rank? | 38 | 11 | 0.338 | 20 | 0.549 | fora do vocabulário TF-IDF: fre, conseg, rank; termos que puxaram para o id 11: jog |
| Qual o tempo de round no competitivo? | 59 | 28 | 0.279 | 28 | 0.618 | fora do vocabulário TF-IDF: round |
| Como o time CT ganha o round? | 62 | 53 | 0.325 | 53 | 0.578 | nenhum termo em comum com o texto indexado (pergunta+resposta) do id esperado (sinonímia/vocabulário); fora do vocabulário TF-IDF: round; tem sigla: ct; termos que puxaram para o id 53: ct, ganh, tim |
| o jogo fica dando crash, oq eu faço | 87 | 81 | 0.227 | 31 | 0.511 | fora do vocabulário TF-IDF: dand; termos que puxaram para o id 81: fac |
<!-- /AUTO:casos -->

### 6.1 Hipóteses explicativas (ligadas aos conceitos da disciplina)

Houve casos nos dois sentidos: 9 perguntas em que só a v2 pôs a resposta certa em 1º
lugar e 4 em que só a v1 pôs. Abaixo, três de cada lado e três em que as duas erraram.

**Onde a v2 ganhou**

1. **"Minha patente da mirage desapareceu do nada" (esperado: id 34) — sinonímia.** A base
   pergunta "Por que minha patente sumiu?". Para o TF-IDF, "desapareceu" e "sumiu" são
   dimensões diferentes, sem nenhuma relação; sobrou "patente", que aparece em várias
   perguntas, e a v1 escolheu o id 35 ("O suporte pode alterar minha patente?", 0,235). A v2
   deu 0,659 ao id 34. Os embeddings seguem a **hipótese distribucional**: palavras usadas em
   contextos parecidos ficam com vetores parecidos, então "desapareceu" fica perto de "sumiu".
2. **"Que jogo é o Counter Strike 2, quem fez e quando saiu?" (id 1) — tokenização e
   polissemia.** A base escreve "Counter-Strike" com hífen, que o tokenizador mantém como um
   token só; o usuário escreveu separado, e "count"/"strik" ficaram fora do vocabulário. O
   único termo que sobrou com peso foi "saiu", e ele puxou a v1 para o id 26 ("A Overpass
   **saiu** do Premier"). É o mesmo verbo com outro sentido: aqui "saiu" é "foi lançado". A v2
   olha a frase inteira (é isso que a **atenção** faz: cada token olha os outros) e deu 0,779
   ao id 1.
3. **"meus bans antigos do csgo passaram pro cs2 tambem?" (id 5) — stemming e
   tokenização.** "bans" virou `bam` e não casou com `ban`; "csgo" escrito junto não casou com
   "CS:GO", que vira `cs` + `go`. A v1 ficou com "passaram" e foi para o id 48. A v2 não
   depende de os termos coincidirem letra por letra e deu 0,696 ao id 5.

Também apareceram "quando acaba a **season** o meu rating zera?" ("season" fora do
vocabulário; a base diz "temporada") e "**não** consigo entrar no premier, o botão tá cinza".
Nos dois a v2 pôs o id certo em 1º, mas com escore abaixo do limiar, então a resposta final foi
"não entendi" (ver a leitura da seção 5).

**Onde a v1 ganhou**

1. **"Comprei uma chave de caixa por engano, consigo estorno?" (id 90) — termo exato.**
   "estorno" está fora do vocabulário, mas "chave", "caixa" e "comprei" casam com o id 90
   ("reembolso de caixas, chaves ou passes **comprados**"). A v2 foi para o id 97 ("como
   reverto as trocas?"): pegou a intenção geral, desfazer uma transação, e perdeu o objeto
   específico. Detalhe: a v1 acertou o ranking, mas o escore (0,304) ficou abaixo do limiar
   dela (0,33), então a resposta final também foi rejeitada.
2. **"Onde eu compro o prime?" (id 40) — duas perguntas vizinhas.** A base tem "Como compro o
   Status Prime?" (40) e "Quanto custa o Status Prime?" (41). A v1 casou "compro" com a pergunta
   do id 40, que usa o mesmo verbo. Para a v2 as duas frases tratam do mesmo assunto (comprar o Prime) e o 41 ficou
   um pouco à frente; o 40 veio em 2º, o que aparece no MRR.
3. **"Como funcionam os convites da Valve pros times irem ao Major?" (id 100) — palavra que só
   está na resposta.** "convites" não aparece em nenhuma pergunta da base, só na resposta do id
   100 ("define os **convites** para os Majors"). Indexar pergunta + resposta (3.1) deu esse
   acerto à v1. A v2 foi para o id 99 ("Como é o formato de um Major?"), em que o assunto
   "Major" domina o vetor.

Nenhum caso de "v1 ganha" envolveu sigla: nas perguntas com VAC, AWP e CS2 as duas versões
acertaram juntas. O padrão que apareceu foi outro: a v1 ganhou quando havia um substantivo
exato que só um documento usa, e a v2 trocou o assunto específico por um vizinho de tema.

**Onde as duas erraram**

- **"Como o time CT ganha o round?" (id 62).** A base diz "contra-terroristas" e "rodada";
  não há nenhum termo em comum com o id 62. O dicionário de siglas protege "CT", mas não sabe
  que CT = contra-terrorista. As duas foram para o id 53, cuja resposta cita "CT" e "vencer
  uma rodada". Uma melhoria seria expandir as siglas para a forma por extenso.
- **"jogador free consegue ter rank?" (id 38).** "free" quer dizer "sem Prime", que é
  conhecimento do domínio. A v2 deixou o id 38 só em 19º lugar.
- **"Ganhar no competitivo sobe meu rating do premier?" (id 16).** A base tem três perguntas
  vizinhas sobre CS Rating (14, 15 e 16). A v1 foi para o 15, cuja resposta diz que o rating
  "**sobe** com vitórias"; a v2 foi para o 14. A diferença que importa (Competitivo × Premier)
  é uma palavra em frases quase iguais.


---

## 7. Decisões adicionais (não listadas no enunciado)

1. **Ranking por resposta (max-pooling por grupo)** para perguntas com resposta repetida (3.4).
2. **Dicionário de siglas extraído da base** + plural de sigla + detecção de CAPS LOCK (2.1).
3. **Ordem stemming → remoção de acentos**, porque as regras do RSLP usam sufixos acentuados.
4. **Objetivo balanceado para o limiar** (J) e **meio da maior faixa empatada** (3.2).
5. **Métrica extra `acerto_final`**, que junta ranking e limiar no que o usuário vê.
6. **Entradas estranhas:** `None`, número, string vazia, só pontuação, emoji → resposta de
   rejeição com escore 0; textos com mais de 1000 caracteres são cortados (o modelo já
   trunca em ~128 tokens, e o TF-IDF não precisa de mais). Testado em
   `tests/test_normalizacao_e_robustez.py`. O `chat.py` ainda captura qualquer exceção
   inesperada para nunca cair.
7. **Perguntas em inglês** são tratadas como fora do escopo (a base é em português).
   A v2, por ser multilíngue, pode "entender" e responder uma pergunta em inglês sobre
   CS2 — o que seria até desejável; registramos como limitação/oportunidade.
8. **Níveis de confiabilidade das fontes:** oficiais (Valve) × especializadas. Respostas
   com números que mudam por atualização (pool de mapas, Majors) têm a data na resposta.
9. **Cache com hash do conteúdo**, para não usar embeddings velhos se a base mudar.
10. **Tokenizador do professor (AFD)** como alternativa medida ao NLTK (2.2).
11. **Dicionário de abreviações de chat** nas duas versões (2.1), medido na validação (4.1 e
    5.1) e num teste específico de linguagem informal (8.1).

---

## 8. Limites do sistema

- Recuperação não **gera** resposta: se a dúvida não está na base, o melhor que o bot
  faz é dizer que não entendeu.
- Perguntas com duas dúvidas ("quanto custa o prime e como compro?") só recebem uma resposta.
- A base envelhece (mapas, Majors, preços) e precisa de manutenção.
- O limiar é único para todas as perguntas; perguntas muito curtas ou com sigla
  poderiam merecer limiares diferentes.

### 8.1 Linguagem informal: acentos e abreviações (`src/teste_informal.py`)

O enunciado pede que o teste varie o registro (formal/informal) e tenha erros de digitação.
Para medir isso de forma isolada, escrevemos 12 dúvidas da base de três jeitos, mudando só a
escrita: **formal** (com acentos), **sem acento** e **abreviada** (sem acento e com "vc",
"pq", "q", "qnto", "pra", "tb", "eh", "n"...). Cada versão rodou sem e com o dicionário de
abreviações (2.1). Esse conjunto não foi usado para escolher limiar, modelo nem pipeline.

Respostas finais certas (já com o limiar), em 12 perguntas por forma:

| forma | v1 sem dicionário | v1 com dicionário | v2 sem dicionário | v2 com dicionário |
|---|---|---|---|---|
| formal | 11 | 11 | 11 | 11 |
| sem acento | 11 | 11 | 9 | 9 |
| abreviada | 9 | 10 | 5 | 8 |

**Leitura.**

- **Falta de acento:** a v1 não perdeu nenhuma pergunta, porque o pipeline v0 remove os
  acentos. Mas apareceu um efeito colateral da ordem stemming → acentos (decisão 3 da seção 7):
  a mesma palavra com e sem acento pode virar radicais diferentes, porque o RSLP só reconhece
  alguns sufixos na forma acentuada. Exemplos: "necessárias" → `necessar`, mas "necessarias" →
  `necess`; "verificação" → `verific`, mas "verificacao" → `verificaca`; "possível" →
  `possivel`, mas "possivel" → `possi`. Isso não mudou nenhum 1º lugar aqui, mas baixou o
  escore (id 14: 0,541 → 0,447; id 87: 0,584 → 0,493). A v2 perdeu uma pergunta no ranking e
  outra pelo limiar (id 50: escore 0,539, logo abaixo de 0,54).
- **Abreviações sem o dicionário:** a v1 perdeu 2 de 12 e a v2 perdeu 6. Na v1, "qntas" virou
  `qnt` e não casou com `quant`. A v2 sofreu mais porque foi treinada com frases bem escritas:
  as abreviações viram sub-palavras que ela quase não viu, o que desloca o vetor da frase toda
  ("vc consegue ver seu trust factor?" foi para o id 80).
- **Com o dicionário:** a v1 recuperou "uma partida competitiva tem qntas rodadas no max?"
  (`qnt` → `quant`), e a v2 subiu de 5 para 8. Ela recuperou essa mesma pergunta e a do Trust
  Factor, e duas que já estavam em 1º lugar passaram do limiar (id 50: 0,506 → 0,560; id 56:
  0,544 → 0,623).
- **O que o dicionário não resolve:** gíria que não é abreviação ("ban" no lugar de
  "banimento": "da pra recorrer de um ban do vac?" continua indo para o id 62 na v2, com
  escore 0,32), abreviação fora da lista ("arqvs") e as perguntas que já erravam na forma
  formal (id 40 na v2, id 90 nas duas).
- **Conclusão:** o sistema entende bem perguntas sem acento. Com o dicionário, entende a maior
  parte das perguntas abreviadas: 10 de 12 na v1 e 8 de 12 na v2. Neste teste a v1 foi mais
  robusta que a v2, o contrário do teste principal.

---

## 9. Bônus – ajuste fino eficiente com LoRA e DoRA (`src/bonus_lora_dora.py`)

Escolhemos o caminho LoRA/DoRA da Seção 4.4 do enunciado (não exige chave de API e roda em CPU).

**Tarefa de demonstração.** Classificar a **categoria** de uma pergunta (11 categorias da
coluna `categoria` da base: premier, punicoes, economia, trocas_reembolso...). Treino: as
100 perguntas da base. Avaliação: as perguntas do escopo de `validacao.csv` + `teste.csv`
(a categoria vem do `id_esperado`). Nenhuma decisão do bônus é tomada olhando essa avaliação.

**Modelo.** O mesmo encoder da v2 (`paraphrase-multilingual-MiniLM-L12-v2`), com uma camada
de classificação nova no final (`AutoModelForSequenceClassification`).

**Mecanismo.**

- **Ajuste fino completo** mudaria todos os ~118 milhões de pesos do modelo.
- **LoRA** congela o peso original `W` de cada projeção de atenção (`query` e `value`) e
  aprende só um "desvio" de posto baixo: `W' = W + (alpha/r)·B·A`, com `A` de tamanho
  `r × k` e `B` de tamanho `d × r`. Com `r = 8` e `d = k = 384`, em vez de 147.456 números
  por matriz treinamos 6.144. A hipótese é que a mudança necessária para uma tarefa nova
  "cabe" em poucas direções (posto baixo).
- **DoRA** (`use_dora=True`) separa cada peso em **magnitude** e **direção**:
  `W' = m · (W + B·A) / ||W + B·A||`. A parte LoRA ajusta a direção e o vetor `m` (um número
  por linha da matriz) ajusta a magnitude. Custa poucos parâmetros a mais que a LoRA e
  costuma se comportar de forma mais parecida com o ajuste fino completo.
- Como referência, treinamos também **só a camada de classificação** (encoder congelado).

Hiperparâmetros (iguais para os três, semente 42): 12 épocas, lote 16, AdamW com taxa 2e-3,
`r = 8`, `alpha = 16`, dropout 0,1.

**Resultado** (gerado por `python src/bonus_lora_dora.py`):

<!-- AUTO:bonus -->
| método | parâmetros treináveis | parâmetros totais | % treinável | acurácia (val+teste do escopo) | tempo de treino (s) |
|---|---|---|---|---|---|
| cabeça apenas | 4235 | 117657995 | 0.004 | 0.310 | 16.993 |
| LoRA | 151691 | 117809686 | 0.129 | 0.707 | 35.299 |
| DoRA | 160907 | 117818902 | 0.137 | 0.724 | 41.934 |

Categoria prevista para perguntas que não estão na base:

| pergunta | cabeça apenas | LoRA | DoRA |
|---|---|---|---|
| quanto custa o prime? | punicoes | economia | economia |
| levei ban por sair no meio da partida | punicoes | punicoes | punicoes |
| em quanto tempo a bomba explode? | regras_rodada | regras_rodada | regras_rodada |
| como peço reembolso de uma chave? | punicoes | trocas_reembolso | trocas_reembolso |
| quando vai ser o próximo major? | premier | esports | esports |
| como denuncio um hacker? | punicoes | punicoes | punicoes |
<!-- /AUTO:bonus -->

**Leitura.**

- **Fração treinada.** A LoRA treina 151.691 de 117,8 milhões de parâmetros (0,13%): são
  147.456 dos adaptadores (12 camadas × 2 matrizes × 6.144) mais os 4.235 da camada de
  classificação. A DoRA treina 160.907 (0,14%); os 9.216 a mais são os vetores de magnitude
  (12 × 2 × 384, um número por linha de cada matriz adaptada), exatamente como o mecanismo
  prevê.
- **Acurácia.** Nas 58 perguntas do escopo de validação + teste, a LoRA acertou a categoria
  em 70,7% (41) e a DoRA em 72,4% (42), contra 31,0% (18) treinando só a camada final. Isso
  fica bem acima do acaso (1/11 ≈ 9%) e de chutar sempre a categoria mais comum da base
  (punicoes, 21%). A diferença entre DoRA e LoRA é de uma pergunta, então não dá para dizer
  que uma é melhor que a outra aqui; a diferença para a camada final sozinha é clara.
- **Custo.** Em CPU, cada método treina em menos de 2 minutos, e a camada final sozinha é a mais rápida. Os segundos exatos variam bastante de uma execução para outra (ver a última coluna da tabela).
- **Perguntas novas.** A LoRA e a DoRA acertaram 4 das 5 categorias que esperávamos. O erro
  foi "quanto custa o prime?", classificada como `economia` em vez de `prime_itens`:
  provavelmente porque "quanto" aparece em quase todas as perguntas de economia da base
  ("Quanto dinheiro...", "Quanto o time ganha..."). Só com a camada final, o modelo jogou
  quatro das cinco em `punicoes`, a categoria mais frequente.

O enunciado não exige ganho de desempenho. O que a demonstração mostra é que dá para adaptar
o modelo a uma tarefa nova treinando uma fração mínima dos pesos, em CPU, e que cada tarefa
vira um conjunto pequeno de adaptadores sobre o mesmo modelo base.

