import json
from pathlib import Path
import struct
import tempfile
import tracemalloc
import unittest

from esp32_sd.sd_index import CHUNK, RECORD, SDIndex, export_index, tokenize


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.target = self.root / 'index'
        self.texts = ['Berlin: Berlin ist die Hauptstadt von Deutschland.',
                      'Hamburg: Hamburg ist eine Stadt an der Elbe.',
                      'München: München liegt in Bayern. Grüße aus München!']
        words, titles = {}, {}
        for i, text in enumerate(self.texts):
            for word in set(tokenize(text)):
                words.setdefault(word, []).append(i)
            for word in set(tokenize(text.split(':')[0])):
                titles.setdefault(word, []).append(i)
        meta = {'n_chunks': len(self.texts)}
        for field, filename, entries in [('woerter', 'postings.npy', words), ('titel', 'titel_postings.npy', titles)]:
            data = []
            meta[field] = {}
            for word, ids in entries.items():
                meta[field][word] = [len(data), len(ids)]
                data.extend(ids)
            (self.source / filename).write_bytes(struct.pack('<' + 'I' * len(data), *data))
        (self.source / 'vokabular.json').write_text(json.dumps(meta))
        (self.source / 'chunks.txt').write_text('\n'.join(self.texts) + '\n', encoding='utf-8')
        export_index(self.source, self.target)

    def test_ranking_utf8_unknown(self):
        with SDIndex(self.target) as index:
            self.assertEqual(index.suche('Wo liegt München?')[0][1], self.texts[2])
            self.assertEqual(index.suche('Was ist Berlin?')[0][1], self.texts[0])
            self.assertEqual(index.suche('xyzzy unfindbar'), [])
            self.assertEqual(index.text(2), self.texts[2])

    def test_limits_and_memory(self):
        tracemalloc.start()
        try:
            with SDIndex(self.target) as index:
                index.suche('Berlin Hamburg München')
                with self.assertRaises(ValueError):
                    index.suche('a' * 4097)
                with self.assertRaises(ValueError):
                    index.suche('Berlin', 31)
                with self.assertRaises(IndexError):
                    index.text(3)
            self.assertLess(tracemalloc.get_traced_memory()[1], 1024 * 1024)
        finally:
            tracemalloc.stop()

    def test_truncated_vocabulary(self):
        p = self.target / 'vocab.bin'
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            SDIndex(self.target)

    def test_bad_posting(self):
        p = self.target / 'postings.bin'
        p.write_bytes(struct.pack('<I', 999) + p.read_bytes()[4:])
        with SDIndex(self.target) as index:
            first = RECORD.unpack((self.target / 'vocab.bin').read_bytes()[:RECORD.size])
            with self.assertRaises(ValueError):
                index._posting('postings.bin', 0)

    def test_corrupt_compression(self):
        p = self.target / 'chunks.idx'
        p.write_bytes(CHUNK.pack(0, 5, 70000) + p.read_bytes()[CHUNK.size:])
        with SDIndex(self.target) as index:
            with self.assertRaises(ValueError):
                index.text(0)

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            export_index(self.source, self.target)


if __name__ == '__main__':
    unittest.main()
