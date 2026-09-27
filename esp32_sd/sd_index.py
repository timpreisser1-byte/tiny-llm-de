"""Desktop exporter and bounded-working-set reference reader, not firmware."""
import argparse
import heapq
import json
import math
from pathlib import Path
import re
import shutil
import struct
import zlib

RECORD = struct.Struct('<80sQIQI')
CHUNK = struct.Struct('<QII')
U32 = struct.Struct('<I')
MAX_CHUNK = 65536
MAX_TERMS = 32
STOP = set('der die das den dem des ein eine einen einem einer eines und oder aber ist sind war waren wird werden wurde wurden hat habe haben hatte hatten sich nicht auch noch schon nur sehr mehr als wie was wer wo bei mit von vom für fuer auf aus dass weil wenn dann durch über ueber unter zwischen nach vor seit bis zum zur ins man ihm ihn ihr ihre sein seine sowie etwa rund jahr jahre jahren teil zwei drei erste ersten heute wann warum wieso weshalb welche welcher welches welchem welchen viel viele vieles lang lange laenger länger hoch hohe hohen tief tiefe gross groß grosse große alt alte weit weite schwer schwere breit breite nenne sage sag gib zeige heisst heißt bedeutet'.split())


def tokenize(text):
    return [w for w in re.findall(r'[a-zäöüß0-9]{3,20}', text.lower()) if w not in STOP]


def export_index(source, output):
    """One-time host conversion; the exporter intentionally loads source JSON."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (source / 'vokabular.json').open(encoding='utf-8') as f:
        meta = json.load(f)
    words, titles = meta['woerter'], meta.get('titel', {})
    with (output / 'vocab.bin').open('wb') as f:
        keys = sorted(set(words) | set(titles), key=lambda w: w.encode('utf-8'))
        for word in keys:
            key = word.encode('utf-8')
            if len(key) >= 80 or b'\0' in key:
                raise ValueError('Invalid vocabulary key')
            start, count = words.get(word, (0, 0))
            tstart, tcount = titles.get(word, (0, 0))
            f.write(RECORD.pack(key, start, count, tstart, tcount))
    for old, new in [('postings.npy', 'postings.bin'), ('titel_postings.npy', 'titles.bin')]:
        if (source / old).exists():
            shutil.copyfile(source / old, output / new)
        else:
            (output / new).touch()
    n = 0
    with (source / 'chunks.txt').open('rb') as src, (output / 'chunks.bin').open('wb') as dst, (output / 'chunks.idx').open('wb') as idx:
        while True:
            raw = src.readline(MAX_CHUNK + 1)
            if not raw:
                break
            if len(raw) > MAX_CHUNK:
                raise ValueError('Source paragraph exceeds 64 KiB limit')
            raw.decode('utf-8')
            packed = zlib.compress(raw, 6)
            idx.write(CHUNK.pack(dst.tell(), len(packed), len(raw)))
            dst.write(packed)
            n += 1
    if n != meta['n_chunks']:
        raise ValueError('Source paragraph count mismatch')
    manifest = {'format': 'esp32-sd-index-v1', 'n_chunks': n, 'n_words': len(keys), 'max_chunk_bytes': MAX_CHUNK, 'byte_order': 'little', 'record_bytes': RECORD.size}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


class SDIndex:
    """Binary-search vocabulary and streaming posting merge; no index-wide load."""
    def __init__(self, path):
        self.path = Path(path)
        meta = json.loads((self.path / 'manifest.json').read_text())
        if meta['format'] != 'esp32-sd-index-v1':
            raise ValueError('Unsupported index')
        self.n = meta['n_chunks']
        self.words = meta['n_words']
        if not isinstance(self.n, int) or not isinstance(self.words, int) or self.n < 0 or self.words < 0:
            raise ValueError('Invalid index counts')
        for name, expected in [('vocab.bin', self.words * RECORD.size), ('chunks.idx', self.n * CHUNK.size)]:
            if (self.path / name).stat().st_size != expected:
                raise ValueError('Invalid file size: ' + name)
        for name in ['postings.bin', 'titles.bin']:
            if (self.path / name).stat().st_size % 4:
                raise ValueError('Invalid posting file size')
        self.files = {name: (self.path / name).open('rb') for name in ['vocab.bin', 'postings.bin', 'titles.bin', 'chunks.bin', 'chunks.idx']}

    def close(self):
        for f in self.files.values():
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def lookup(self, word):
        key = word.encode('utf-8')
        f = self.files['vocab.bin']
        low, high = 0, self.words
        while low < high:
            mid = (low + high) // 2
            f.seek(mid * RECORD.size)
            stored, start, count, ts, tc = RECORD.unpack(f.read(RECORD.size))
            stored = stored.rstrip(b'\0')
            if stored == key:
                for name, pos, length in [('postings.bin', start, count), ('titles.bin', ts, tc)]:
                    if length > self.n or (pos + length) * 4 > (self.path / name).stat().st_size:
                        raise ValueError('Invalid posting range')
                return start, count, ts, tc
            if stored < key:
                low = mid + 1
            else:
                high = mid
        return None

    def text(self, number):
        if not 0 <= number < self.n:
            raise IndexError(number)
        idx = self.files['chunks.idx']
        idx.seek(number * CHUNK.size)
        offset, packed_len, raw_len = CHUNK.unpack(idx.read(CHUNK.size))
        if raw_len > MAX_CHUNK or packed_len > MAX_CHUNK + 1024:
            raise ValueError('Invalid chunk size')
        f = self.files['chunks.bin']
        f.seek(offset)
        dec = zlib.decompressobj()
        raw = dec.decompress(f.read(packed_len), MAX_CHUNK + 1)
        if len(raw) != raw_len or not dec.eof or dec.unused_data:
            raise ValueError('Invalid compressed paragraph')
        return raw.decode('utf-8').strip()

    def _posting(self, filename, position):
        f = self.files[filename]
        f.seek(position * 4)
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError('Truncated posting')
        number = U32.unpack(raw)[0]
        if number >= self.n:
            raise ValueError('Invalid paragraph number')
        return number

    def suche(self, frage, anzahl=3):
        if not 1 <= anzahl <= 30:
            raise ValueError('anzahl must be between 1 and 30')
        if len(frage) > 4096:
            raise ValueError('Question exceeds 4096 characters')
        terms = set(tokenize(frage))
        if len(terms) > MAX_TERMS:
            raise ValueError('At most 32 search terms supported')
        streams, queue = [], []
        for word in sorted(terms):
            entry = self.lookup(word)
            if not entry:
                continue
            start, count, ts, tc = entry
            rarity = math.log(self.n / (count or tc))
            for filename, position, length, weight in [('postings.bin', start, count, rarity), ('titles.bin', ts, tc, rarity * 4)]:
                if length:
                    i = len(streams)
                    streams.append([filename, position, length, weight])
                    heapq.heappush(queue, (self._posting(filename, position), i))
        candidates = []
        while queue:
            doc = queue[0][0]
            score = 0.0
            while queue and queue[0][0] == doc:
                _, i = heapq.heappop(queue)
                stream = streams[i]
                filename, pos, length, weight = stream
                score += weight
                stream[1] += 1
                stream[2] -= 1
                if stream[2]:
                    following = self._posting(filename, pos + 1)
                    if following <= doc:
                        raise ValueError('Postings must be strictly increasing')
                    heapq.heappush(queue, (following, i))
            candidate = (score, -doc)
            if len(candidates) < 30:
                heapq.heappush(candidates, candidate)
            elif candidate > candidates[0]:
                heapq.heapreplace(candidates, candidate)
        ranked = []
        for score, negdoc in candidates:
            text = self.text(-negdoc)
            title = text.split(':', 1)[0] if ':' in text[:80] else ''
            titlewords = set(tokenize(title))
            if titlewords:
                score *= 1 + 2 * len(titlewords & terms) / len(titlewords)
            ranked.append((score, -negdoc))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [(score, self.text(doc)) for score, doc in ranked[:anzahl]]


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='action', required=True)
    exp = subs.add_parser('export')
    exp.add_argument('--source', required=True)
    exp.add_argument('--output', required=True)
    search = subs.add_parser('search')
    search.add_argument('--index', required=True)
    search.add_argument('question')
    args = parser.parse_args()
    if args.action == 'export':
        print(json.dumps(export_index(args.source, args.output), indent=2))
    else:
        with SDIndex(args.index) as index:
            print(json.dumps(index.suche(args.question), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
