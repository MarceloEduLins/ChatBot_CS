"""Testes do lexer AFD (adaptado do analisador_lexico.c)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lexer_afd import TipoToken, classificar, reconhece_numero, tokenizar  # noqa: E402


class TestAFDNumeros(unittest.TestCase):
    def test_mesmos_casos_do_codigo_em_c(self):
        # (entrada, comprimento aceito) - mesmo comportamento do testar_afd() em C
        casos = {"123": 3, "-45.67": 6, "3.14e-10": 8, "abc": 0,
                 "12.": 2, ".5": 0, "1e10": 4, "7": 1}
        for entrada, esperado in casos.items():
            self.assertEqual(reconhece_numero(entrada, 0), esperado, entrada)

    def test_afd_equivale_a_regex(self):
        """Aula 02: toda linguagem regular tem uma MEF e uma Regex equivalentes
        (MEF ≡ Regex). Aqui conferimos isso para o AFD de números, em 3000
        strings aleatórias: o AFD aceita a string inteira exatamente quando a
        expressão regular também aceita, e o maior prefixo é o mesmo."""
        import random
        import re
        regex = re.compile(r"[+-]?[0-9]+([.,][0-9]+)*([eE][+-]?[0-9]+)?")
        rng = random.Random(42)
        for _ in range(3000):
            s = "".join(rng.choice("0123456789.,+-eEa") for _ in range(rng.randint(1, 8)))
            aceita_afd = reconhece_numero(s, 0) == len(s)
            aceita_regex = regex.fullmatch(s) is not None
            self.assertEqual(aceita_afd, aceita_regex, s)
            prefixo = regex.match(s)
            self.assertEqual(reconhece_numero(s, 0), prefixo.end() if prefixo else 0, s)

    def test_virgula_decimal(self):
        self.assertEqual(reconhece_numero("14,99", 0), 5)
        self.assertEqual(reconhece_numero("16.000", 0), 6)


class TestLexer(unittest.TestCase):
    def tipos(self, texto):
        return [(t.tipo, t.lexema) for t in tokenizar(texto)]

    def test_siglas_numeros_e_palavras(self):
        toks = self.tipos("O Prime custa US$ 14,99 no CS2")
        self.assertIn((TipoToken.SIGLA, "CS2"), toks)
        self.assertIn((TipoToken.SIGLA, "US"), toks)
        self.assertIn((TipoToken.NUMERO, "14,99"), toks)
        self.assertIn((TipoToken.PALAVRA, "Prime"), toks)
        self.assertIn((TipoToken.PONTUACAO, "$"), toks)

    def test_maior_prefixo(self):
        self.assertEqual(self.tipos("2v2"), [(TipoToken.PALAVRA, "2v2")])
        self.assertEqual(self.tipos("pick-ban"), [(TipoToken.PALAVRA, "pick-ban")])
        self.assertEqual(self.tipos("cs-")[0], (TipoToken.PALAVRA, "cs"))

    def test_nao_trava_com_emoji(self):
        toks = self.tipos("gg 😀 ez")
        self.assertEqual(toks[0], (TipoToken.PALAVRA, "gg"))
        self.assertEqual(toks[1][0], TipoToken.INVALIDO)
        self.assertEqual(toks[2], (TipoToken.PALAVRA, "ez"))

    def test_plural_de_sigla(self):
        self.assertEqual(classificar("CTs"), TipoToken.SIGLA)
        self.assertEqual(classificar("VAC"), TipoToken.SIGLA)
        self.assertEqual(classificar("Vac"), TipoToken.PALAVRA)
        self.assertEqual(classificar("A"), TipoToken.PALAVRA)   # 1 letra não é sigla

    def test_acentos_sao_letras(self):
        self.assertEqual(self.tipos("patente"), [(TipoToken.PALAVRA, "patente")])
        self.assertEqual(self.tipos("não"), [(TipoToken.PALAVRA, "não")])


if __name__ == "__main__":
    unittest.main()
