"""
chatbot_tfidf.py - v1: recuperação com TF-IDF + similaridade de cossenos.

Fluxo: pergunta -> normalizar (v0) -> vetor TF-IDF -> cosseno com cada
pergunta da base -> ranking -> limiar de rejeição.

TF-IDF do scikit-learn (padrão smooth_idf=True, norm='l2'):
    idf(t)   = ln((1 + N) / (1 + df(t))) + 1
    tfidf    = tf(t, d) * idf(t), depois o vetor é normalizado (norma L2)
Como os vetores já saem com norma 1, o cosseno vira só o produto escalar,
mas mesmo assim usamos a nossa função completa (similaridade.cosseno).
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from assistente import AssistenteBase
from normalizacao import ConfigNormalizacao, Normalizador
from similaridade import cosseno

LIMIAR_PADRAO_V1 = 0.30   # só vale até rodar a avaliação (que escolhe o limiar na validação)


class AssistenteTFIDF(AssistenteBase):
    nome = "v1"

    def __init__(self, base, limiar=LIMIAR_PADRAO_V1, campo="pergunta", config=None):
        super().__init__(base, limiar)
        self.campo = campo
        self.config = config or ConfigNormalizacao()
        self.normalizador = Normalizador(self.config, siglas=base.siglas)
        # analyzer recebe o texto cru e devolve a lista de tokens do pipeline v0
        self.vetorizador = TfidfVectorizer(analyzer=self.normalizador.normalizar)
        self.matriz = self.vetorizador.fit_transform(base.textos(campo))

    def _escores(self, texto):
        q = self.vetorizador.transform([texto])
        return cosseno(q, self.matriz)

    def vocabulario(self):
        return sorted(self.vetorizador.vocabulary_)


if __name__ == "__main__":
    from base import carregar_base
    assistente = AssistenteTFIDF(carregar_base())
    for p in ["quanto tempo dura o cooldown?", "qual a senha do wifi?"]:
        r = assistente.responder(p)
        print(p, "->", r.id, round(r.escore, 3), r.resposta[:80])
