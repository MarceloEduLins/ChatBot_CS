"""Confere o nosso cosseno (NumPy) contra o do scikit-learn."""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from base import carregar_base  # noqa: E402
from similaridade import cosseno, cosseno_matriz  # noqa: E402


class TestCosseno(unittest.TestCase):
    def test_vetores_aleatorios(self):
        rng = np.random.default_rng(42)
        A = rng.normal(size=(7, 20))
        B = rng.normal(size=(30, 20))
        np.testing.assert_allclose(cosseno_matriz(A, B), cosine_similarity(A, B), atol=1e-10)

    def test_tfidf_real_da_base(self):
        """Mesma matriz esparsa que a v1 usa (aqui com o analisador padrão do
        sklearn, para o teste não depender do NLTK)."""
        base = carregar_base()
        vet = TfidfVectorizer()
        D = vet.fit_transform(base.textos("pergunta"))
        Q = vet.transform(["quanto custa o prime?", "o que é o VAC", "blablabla"])
        np.testing.assert_allclose(cosseno_matriz(Q, D), cosine_similarity(Q, D), atol=1e-10)

    def test_vetor_nulo_da_zero(self):
        D = np.array([[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(cosseno(np.zeros(2), D), [0.0, 0.0])
        D_com_nulo = np.array([[0.0, 0.0], [3.0, 4.0]])
        np.testing.assert_allclose(cosseno([3.0, 4.0], D_com_nulo), [0.0, 1.0])

    def test_valores_conhecidos(self):
        self.assertAlmostEqual(cosseno([1, 0], [[1, 0]])[0], 1.0)
        self.assertAlmostEqual(cosseno([1, 0], [[0, 1]])[0], 0.0)
        self.assertAlmostEqual(cosseno([1, 0], [[-1, 0]])[0], -1.0)
        self.assertAlmostEqual(cosseno([1, 1], [[1, 0]])[0], 1 / np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
