"""Pipeline v0 e robustez do assistente (precisam do NLTK e dos seus dados)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from assistente import MSG_REJEICAO  # noqa: E402
from base import carregar_base  # noqa: E402

try:
    import nltk  # noqa: F401
    from normalizacao import normalizar
    normalizar("teste")          # força o carregamento dos recursos
    TEM_NLTK = True
except Exception:                # sem NLTK ou sem internet para baixar os dados
    TEM_NLTK = False


class TestAbreviacoes(unittest.TestCase):
    """Não precisa do NLTK: a expansão é feita no texto, antes da tokenização."""

    def test_expande(self):
        from normalizacao import expandir_abreviacoes
        self.assertEqual(expandir_abreviacoes("qnto eh q custa o prime?"),
                         "quanto é que custa o prime?")
        self.assertEqual(expandir_abreviacoes("vc tb joga? pq n"), "você também joga? por que não")
        self.assertEqual(expandir_abreviacoes("Pra que serve? PQ?"), "para que serve? por que?")

    def test_nao_mexe_em_palavra_maior_nem_sigla(self):
        from normalizacao import expandir_abreviacoes
        texto = "praia 3d pick-ban de_dust2 CS2 VAC to num"
        self.assertEqual(expandir_abreviacoes(texto), texto)


@unittest.skipUnless(TEM_NLTK, "NLTK (e punkt_tab/stopwords/rslp) não disponível")
class TestNormalizacao(unittest.TestCase):
    def test_vazio(self):
        self.assertEqual(normalizar(""), [])
        self.assertEqual(normalizar("   "), [])
        self.assertEqual(normalizar("?!..."), [])

    def test_negacao_mantida(self):
        self.assertIn("nao", normalizar("Não consigo jogar"))
        self.assertNotIn("nao", normalizar("Não consigo jogar", manter_negacao=False))

    def test_marcar_negacao(self):
        toks = normalizar("não consigo jogar", marcar_negacao=True)
        self.assertTrue(any(t.startswith("nao_") for t in toks), toks)

    def test_sigla_protegida(self):
        base = carregar_base()
        self.assertIn("vrs", normalizar("O que é o VRS?", siglas=base.siglas))
        self.assertIn("vrs", normalizar("o que é o vrs", siglas=base.siglas))  # minúsculas

    def test_etapas_desligaveis(self):
        frase = "Quais são as patentes do jogo?"
        self.assertNotEqual(normalizar(frase), normalizar(frase, stemming=False))
        self.assertNotEqual(normalizar(frase), normalizar(frase, remover_stopwords=False))
        self.assertIn("são", normalizar(frase, remover_acentos=False, stemming=False,
                                        remover_stopwords=False))

    def test_tokenizador_afd(self):
        self.assertEqual(normalizar("Quanto custa o Prime?", tokenizador="afd"),
                         normalizar("Quanto custa o Prime?"))


@unittest.skipUnless(TEM_NLTK, "NLTK não disponível")
class TestRobustezV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from chatbot_tfidf import AssistenteTFIDF
        cls.v1 = AssistenteTFIDF(carregar_base())

    def test_entradas_estranhas_nao_travam(self):
        entradas = ["", "   ", "???", "😀😀😀", "a" * 20000, None, 12345,
                    "What is the weather today?", "SELECT * FROM users;", "\n\t"]
        for e in entradas:
            r = self.v1.responder(e)
            self.assertIsInstance(r.resposta, str)
            self.assertIsInstance(r.id, int)
            self.assertIsInstance(r.escore, float)

    def test_vazio_e_rejeitado(self):
        r = self.v1.responder("")
        self.assertEqual((r.id, r.resposta), (-1, MSG_REJEICAO))

    def test_ranking_ordenado_e_sem_resposta_repetida(self):
        r = self.v1.responder("quanto tempo dura um cooldown competitivo")
        escores = [s for _, s in r.ranking]
        self.assertEqual(escores, sorted(escores, reverse=True))
        self.assertEqual(len({i for i, _ in r.ranking}), len(r.ranking))
        self.assertNotIn(65, [i for i, _ in r.ranking])   # 65 é duplicata de 64


if __name__ == "__main__":
    unittest.main()
