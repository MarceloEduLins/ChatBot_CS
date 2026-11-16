# Resultados dos experimentos (gerado por src/avaliacao.py)

Base: 100 pares (95 respostas distintas). Validação: 32 perguntas. Teste: 42 perguntas. Semente: 42.

- Perguntas de `validacao` copiadas literalmente da base: **0** 
- Perguntas de `teste` copiadas literalmente da base: **0** 

## Comparação v1 × v2 (conjunto de teste)

| versao | representacao | campo | limiar | acuracia@1 | mrr | rejeicao_correta | falsa_rejeicao | acerto_final |
|---|---|---|---|---|---|---|---|---|
| v1 | TF-IDF | pergunta+resposta | 0.330 | 0.588 | 0.716 | 0.750 | 0.412 | 0.500 |
| v2 | embeddings (paraphrase-multilingual-mpnet-base-v2) | pergunta+resposta | 0.540 | 0.735 | 0.820 | 1.000 | 0.324 | 0.619 |

## v1 – o que indexar (escolha pela validação)

| versao | campo | limiar_val | val_acuracia@1 | val_mrr | teste_acuracia@1 | teste_mrr |
|---|---|---|---|---|---|---|
| v1 | pergunta | 0.710 | 0.292 | 0.418 | 0.412 | 0.570 |
| v1 | resposta | 0.660 | 0.292 | 0.417 | 0.471 | 0.558 |
| v1 | pergunta+resposta | 0.330 | 0.417 | 0.530 | 0.588 | 0.716 |

Campo escolhido para a v1: **pergunta+resposta**. Limiar escolhido na validação: **0.33**.

## v1 – curva do limiar na validação (a cada 0,05)

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

## Ablação do pipeline v0 (sobre a v1)

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

## v2 – experimentos (campo, normalização, modelo)

| etapa | modelo | campo | normalizar_antes | expandir_abreviacoes | limiar_val | val_acuracia@1 | val_mrr | teste_acuracia@1 | teste_mrr |
|---|---|---|---|---|---|---|---|---|---|
| campo | paraphrase-multilingual-MiniLM-L12-v2 | pergunta | False | True | 0.540 | 0.500 | 0.600 | 0.676 | 0.759 |
| campo | paraphrase-multilingual-MiniLM-L12-v2 | resposta | False | True | 0.540 | 0.292 | 0.440 | 0.471 | 0.588 |
| campo | paraphrase-multilingual-MiniLM-L12-v2 | pergunta+resposta | False | True | 0.480 | 0.500 | 0.638 | 0.706 | 0.777 |
| normalizacao | paraphrase-multilingual-MiniLM-L12-v2 | pergunta+resposta | True | True | 0.540 | 0.292 | 0.433 | 0.412 | 0.515 |
| modelo | paraphrase-multilingual-mpnet-base-v2 | pergunta+resposta | False | True | 0.540 | 0.625 | 0.738 | 0.735 | 0.820 |
| abreviacoes | paraphrase-multilingual-mpnet-base-v2 | pergunta+resposta | False | False | 0.540 | 0.625 | 0.737 | 0.735 | 0.827 |

Escolhas da v2: campo **pergunta+resposta**, normalizar antes: **False**, modelo **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**, limiar **0.54**.

## v2 – curva do limiar na validação (a cada 0,05)

| limiar | escopo_respondido_certo | rejeicao_correta | falsa_rejeicao | J |
|---|---|---|---|---|
| 0.000 | 0.625 | 0.000 | 0.000 | 0.312 |
| 0.050 | 0.625 | 0.000 | 0.000 | 0.312 |
| 0.100 | 0.625 | 0.000 | 0.000 | 0.312 |
| 0.150 | 0.625 | 0.000 | 0.000 | 0.312 |
| 0.200 | 0.625 | 0.125 | 0.000 | 0.375 |
| 0.250 | 0.625 | 0.125 | 0.000 | 0.375 |
| 0.300 | 0.625 | 0.250 | 0.000 | 0.438 |
| 0.350 | 0.625 | 0.625 | 0.000 | 0.625 |
| 0.400 | 0.625 | 0.625 | 0.000 | 0.625 |
| 0.450 | 0.625 | 0.875 | 0.042 | 0.750 |
| 0.500 | 0.625 | 0.875 | 0.042 | 0.750 |
| 0.550 | 0.542 | 1.000 | 0.208 | 0.771 |
| 0.600 | 0.500 | 1.000 | 0.333 | 0.750 |
| 0.650 | 0.375 | 1.000 | 0.542 | 0.688 |
| 0.700 | 0.292 | 1.000 | 0.667 | 0.646 |
| 0.750 | 0.208 | 1.000 | 0.792 | 0.604 |
| 0.800 | 0.125 | 1.000 | 0.875 | 0.562 |
| 0.850 | 0.000 | 1.000 | 1.000 | 0.500 |
| 0.900 | 0.000 | 1.000 | 1.000 | 0.500 |
| 0.950 | 0.000 | 1.000 | 1.000 | 0.500 |
| 1.000 | 0.000 | 1.000 | 1.000 | 0.500 |

## Casos para a análise de erros (teste, perguntas do escopo)

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

_Tempo total: 128.9 s._