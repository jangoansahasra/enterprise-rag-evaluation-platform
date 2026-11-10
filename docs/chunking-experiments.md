# Document processing and chunking experiments

This experiment measures how metadata-preserving chunk construction affects
retrieval quality on the source-verified enterprise IT benchmark.

The pipeline uses the independently collected public GitLab handbook passages
in `data/corpus.json`. It does not construct retrieval documents from benchmark
answers and does not generate synthetic policy content.

## Document-processing pipeline

The pipeline:

1. Loads the public-source registry and independent corpus.
2. Validates source IDs and registered section headings.
3. Normalizes whitespace without rewriting source content.
4. Preserves titles, URLs, publishers, licenses, checked dates, and headings.
5. Generates a stable SHA-256-derived version identifier for each document.
6. Creates searchable chunks with links to their original parent passages.
7. Records a manifest containing the configuration, document versions, source
   count, and chunk count.

Every experimental chunk contains:

- Unique chunk ID
- Parent passage ID
- Source ID
- Section heading
- Source title and URL
- Publisher and license
- Source checked date
- Document version
- Chunking strategy
- Chunk size and overlap
- Chunk index

## Compared configurations

| Configuration | Strategy | Maximum characters | Overlap |
|---|---|---:|---:|
| `heading_aware` | Existing verified sections | Section length | 0 |
| `fixed_80_20` | Fixed-size | 80 | 20 |
| `fixed_120_20` | Fixed-size | 120 | 20 |
| `fixed_160_40` | Fixed-size | 160 | 40 |
| `recursive_80_20` | Natural-boundary preference | 80 | 20 |
| `recursive_120_20` | Natural-boundary preference | 120 | 20 |
| `recursive_160_40` | Natural-boundary preference | 160 | 40 |

BM25 and `top_k=3` remain fixed across experiments. The only changed variable
is the chunking configuration.

Split chunks are evaluated using their parent passage IDs. This allows a
retrieved child chunk to receive credit when it belongs to a verified
supporting passage.

## Run the experiment

```bash
PYTHONPATH=src python scripts/run_chunking_experiments.py
```

The live report is written to:
artifacts/runs/chunking-comparison.json
The versioned baseline is stored at:
```bash
artifacts/baselines/chunking-comparison-v0.1.json
```
## Results
Configuration	Chunks	Average characters	Index growth	Recall@3	MRR	Mean query time
Heading-aware	19	110.5	1.00x	0.8125	0.6875	0.018 ms
Fixed 80/20	37	66.4	1.95x	0.8750	0.6979	0.133 ms
Fixed 120/20	26	86.1	1.37x	0.8125	0.6562	0.055 ms
Fixed 160/40	24	95.7	1.26x	0.8125	0.6771	0.049 ms
Recursive 80/20	37	66.0	1.95x	0.8750	0.6979	0.449 ms
Recursive 120/20	26	85.8	1.37x	0.8750	0.6771	0.048 ms
Recursive 160/40	24	95.6	1.26x	0.8125	0.6458	0.048 ms


## Selected configuration
Fixed-size chunking with 80 characters and 20 characters of overlap produced
the best measured BM25 result:
- Recall@3 increased from 81.25% to 87.50%.
- MRR increased from 0.6875 to 0.6979.
- The index increased from 19 to 37 chunks.
- Average chunk length decreased from 110.5 to 66.4 characters.
- It matched recursive 80/20 retrieval quality with lower measured query time.
The selected configuration therefore improves retrieval coverage at the cost
of approximately 1.95 times as many indexed chunks.
This choice applies to the current small BM25 benchmark. It is an experimental configuration rather than a universal production default.

## Question-level analysis
Two question rankings changed between heading-aware and fixed 80/20 chunking.
Question	Heading-aware rank	Fixed 80/20 rank	Effect
q015	2	3	Relevant evidence remained retrieved but moved down one rank
q016	Missed	3	Previously missed evidence entered the top three


The selected configuration still missed:
- q003: lost or stolen device procedure
- q017: infrastructure incident coordination
These misses indicate that chunking alone does not solve every retrieval
failure. Query formulation, lexical mismatch, dense retrieval, hybrid fusion,
and reranking remain necessary parts of the complete retrieval pipeline.

## Interpretation
Smaller chunks improved Recall@3 by isolating relevant terms that were
previously surrounded by unrelated passage text. The same splitting also
caused one relevant result to move from rank two to rank three.
The 120- and 160-character settings did not consistently outperform the
heading-aware baseline. This shows why chunk size should be selected through
measured experiments instead of by convention.
Fixed and recursive 80/20 produced identical Recall@3 and MRR in this corpus.
The public passages are short, so recursive boundary selection provides limited additional benefit at this scale.

## Limitations
The benchmark contains 16 answerable retrieval cases and 19 original passages.
The results demonstrate the experiment framework but do not establish a
general production optimum.
Query-time differences are fractions of a millisecond and come from one local
run. They are directional and should not be treated as production latency
benchmarks.
Character-based chunk sizes are intentionally small because the collected
public passages range from 42 to 203 characters. Larger source documents will
require additional token-based and semantic chunking experiments.
The experiment currently isolates chunking with BM25. Dense, hybrid, and
reranked chunking comparisons should be evaluated separately because different retrievers can prefer different chunk sizes.