"""
similaridade.py - similaridade de cossenos implementada pelo grupo (NumPy).

    cos(q, d) = (q · d) / (||q|| · ||d||)

O resultado é conferido contra sklearn.metrics.pairwise.cosine_similarity
em tests/test_similaridade.py.
"""

import numpy as np


def cosseno(q, D):
    """Similaridade de cossenos entre o vetor q (n,) e cada linha de D (m, n).

    Devolve um vetor (m,). Se algum dos vetores tiver norma zero (ex.: pergunta
    só com stopwords, que vira o vetor nulo no TF-IDF), a similaridade é 0 em
    vez de dividir por zero.
    """
    if hasattr(q, "toarray"):   # aceita matriz esparsa do scikit-learn
        q = q.toarray()
    if hasattr(D, "toarray"):
        D = D.toarray()
    q = np.asarray(q, dtype=np.float64).ravel()
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))

    produto_escalar = D @ q                                  # q · d para cada d
    normas = np.linalg.norm(D, axis=1) * np.linalg.norm(q)   # ||q|| · ||d||

    sim = np.zeros(D.shape[0])
    ok = normas > 0
    sim[ok] = produto_escalar[ok] / normas[ok]
    return sim


def cosseno_matriz(A, B):
    """Matriz de similaridades (len(A) x len(B)); usada só para conferir
    com o scikit-learn no teste automatizado."""
    A = np.atleast_2d(np.asarray(A.toarray() if hasattr(A, "toarray") else A, dtype=np.float64))
    return np.vstack([cosseno(a, B) for a in A])
