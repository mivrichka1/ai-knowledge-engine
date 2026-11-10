import unittest
from ake_backend.text import TextNormalizer, fixed_chunks


class TextProcessingTests(unittest.TestCase):
    def test_normalization_preserves_structure_and_symbols(self):
        text = 'Заголовок\r\n  •  x²  =  2\x00\r\n\r\nCafe\u0301\t α'
        self.assertEqual(TextNormalizer().normalize(text), 'Заголовок\n• x² = 2\n\nCafé α')

    def test_empty_and_invalid_settings(self):
        self.assertEqual(fixed_chunks('', 4, 1), [])
        for size, overlap in [(0, 0), (2, 2), (2, -1)]:
            with self.assertRaises(ValueError):
                fixed_chunks('hello', size, overlap)

    def test_offsets_overlap_and_unicode(self):
        text = 'Первый α\nsecond  третий x² = 2'
        chunks = fixed_chunks(text, 4, 2)
        self.assertEqual([c.text for c in chunks], ['Первый α\nsecond  третий', 'second  третий x² =', 'x² = 2'])
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.text, text[chunk.start:chunk.end])
            self.assertEqual(chunk.index, i)
        self.assertEqual([c.token_count for c in chunks], [4, 4, 3])

    def test_single_chunk(self):
        self.assertEqual(fixed_chunks('hello world', 20, 2)[0].text, 'hello world')

if __name__ == '__main__':
    unittest.main()
