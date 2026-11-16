"""Checagens da base e dos conjuntos (tamanho, fontes, contaminação)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from base import carregar_base, carregar_perguntas, checar_contaminacao  # noqa: E402


class TestDados(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = carregar_base()
        cls.val = carregar_perguntas("validacao")
        cls.teste = carregar_perguntas("teste")

    def test_tamanho_e_fontes(self):
        self.assertTrue(50 <= len(self.base) <= 100)
        self.assertTrue(self.base.df["fonte"].str.startswith("http").all())
        self.assertFalse(self.base.df[["pergunta", "resposta", "categoria"]].isna().any().any())

    def test_teste_tem_minimo_exigido(self):
        self.assertGreaterEqual((self.teste["id_esperado"] != -1).sum(), 30)
        self.assertGreaterEqual((self.teste["id_esperado"] == -1).sum(), 5)

    def test_ids_esperados_existem(self):
        for df in (self.val, self.teste):
            for i in df["id_esperado"]:
                self.assertTrue(i == -1 or i in self.base.ids, i)

    def test_sem_contaminacao(self):
        self.assertEqual(checar_contaminacao(self.base, self.teste), [])
        self.assertEqual(checar_contaminacao(self.base, self.val), [])

    def test_validacao_e_teste_separados(self):
        comuns = set(self.val["pergunta"].str.lower()) & set(self.teste["pergunta"].str.lower())
        self.assertEqual(comuns, set())

    def test_respostas_duplicadas_viram_um_grupo(self):
        self.assertEqual(self.base.canonico(12), 11)
        self.assertEqual(self.base.canonico(11), 11)
        self.assertEqual(self.base.canonico(-1), -1)

    def test_dicionario_de_siglas(self):
        for s in ["vac", "vrs", "cs2", "mr12", "ct"]:
            self.assertIn(s, self.base.siglas)


if __name__ == "__main__":
    unittest.main()
