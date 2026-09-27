# SD search index, format v1

`sd_index.py` is a desktop exporter and a Python reference reader — not ESP32
firmware. The exporter loads the JSON vocabulary into host RAM; the reader
never loads the vocabulary or the posting lists completely.

From the repository root:

```sh
python -m esp32_sd.sd_index export --source wiki_gestuft --output esp32_sd/wiki_sd_gestuft
python -m esp32_sd.sd_index search --index esp32_sd/wiki_sd_gestuft 'Wo liegt München?'
python -m unittest esp32_sd.test_sd_index
```

The output folder must be new. A failed export can leave an incomplete folder;
only `manifest.json` marks a complete export. Source postings must be in the
existing little-endian `uint32` format, sorted, with unique passage numbers.

All numbers are little-endian. Files:

| File | Layout |
|---|---|
| `manifest.json` | format id, passage and word counts, size limit |
| `vocab.bin` | sorted by UTF-8 bytes, fixed 104-byte records `80s Q I Q I`: zero-padded word, text-posting start/count, title-posting start/count. Starts are `uint32` element positions. |
| `postings.bin`, `titles.bin` | `uint32` passage numbers, strictly increasing per word |
| `chunks.idx` | 16 bytes per passage, `Q I I`: byte offset, compressed length, raw text length |
| `chunks.bin` | independent zlib streams, UTF-8 passages including the original line end |

Binary search reads one vocabulary record per step. The search merges at most
64 posting streams (32 search words, text and title) and keeps only the 30 best
candidates instead of a score table over all passages. Title re-ranking matches
the Python index; on equal scores the passage number decides.

Limits: 4096 characters per question, 32 search words, 30 results, 64 KiB raw
text per passage. The working memory does not depend on the total index size.
The Python interpreter and runtime come on top — this is not a measurement of
ESP32 RAM. The reference reader does small file seeks per posting; SD latency
and a buffered C/C++ reader have to be measured on the board.

The reader checks fixed file sizes, posting ranges and numbers, increasing
postings and bounded decompression. This is not a full integrity check:
unvisited words and postings are only checked on access, and deliberately
altered but still valid content is not detected. Checksums and an atomic swap
of the SD package are still to be added for devices.
