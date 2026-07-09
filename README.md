# Heimdall: a local RAG over Marvel & DC lore

A local Retrieval-Augmented Generation (RAG) pipeline for Marvel and DC wiki lore:

- Corpus: ~50 Marvel and DC character/team articles scraped from the [Marvel](https://marvel.fandom.com) and [DC](https://dc.fandom.com) Fandom wikis via their MediaWiki APIs.
- Models: pluggable LLM providers. Defaults to [Ollama](https://ollama.com) for everything (`nomic-embed-text` for embeddings, `llama3.2:3b` for generation and LLM-as-judge evaluation; no API keys, no cloud). Switch to any OpenAI-compatible API with one environment variable.
- Vector store: a small numpy cosine-similarity index persisted to disk. You can swap in another DB if you want.
- Metrics: built-in evaluation with retrieval metrics (Hit Rate, Precision, Recall, MRR, nDCG), generation metrics (faithfulness, relevance, correctness, token F1, abstention accuracy), and latency stats.

```
Fandom wikis ──rag scrape──> data/corpus/*.md ──rag ingest──> data/index/
                                                                  │
User question ──rag ask──> embed query (LRU cache) ──cosine top-k──┘
                                │
                     low score? └──> live Fandom search → ingest → re-retrieve
                                │
                                └──> prompt with cited context ──llama3.2──> answer + sources
```

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com/download).

```bash
# 1. Start Ollama and pull the models
ollama serve &          # if not already running
ollama pull nomic-embed-text
ollama pull llama3.2:3b

# 2. Install the package
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Check everything is wired up
rag info
```

## Usage

```bash
# Scrape the Marvel/DC corpus from Fandom (~50 articles, a couple of minutes)
rag scrape

# Chunk + embed the corpus into the local vector index
rag ingest

# Ask questions
rag ask "What is the name of Thor's hammer?"
rag ask "How are Superman and Supergirl related?"
rag chat            # interactive loop

# Run the evaluation suite
rag eval                        # all 33 questions
rag eval --limit 8              # quick partial run
rag eval --category alias       # one category only
```

Answers cite the retrieved passages (`[1]`, `[2]`, ...) and the sources table shows which document and section each passage came from. If the answer isn't in the corpus, the model is told to say "I don't know" instead of guessing.

When the best retrieval cosine score is below `RAG_FALLBACK_MIN_SCORE` (default `0.45`), Heimdall searches Marvel/DC Fandom live, fetches the best matching page, appends it to the corpus and vector index, then re-retrieves before generating. Query embeddings are cached in-process (LRU) so repeated questions skip a second embed call.

## Run the whole stack with Docker

[docker-compose.yml](docker-compose.yml) starts Ollama, Qdrant, and the `rag` app together:

```bash
docker compose up -d --build
```

On first start it pulls the models (~2.3 GB into a named volume) and builds the index into Qdrant, so the initial boot takes a few minutes. Then query it:

```bash
docker compose exec rag rag ask "Who trained Doctor Strange?"
docker compose exec rag rag eval --limit 8
docker compose down          # stop (add -v to also wipe model/vector volumes)
```

Notes:
- Images are pinned (`ollama/ollama:0.31.1`, `qdrant/qdrant:v1.18.2`) and the app runs as a non-root user.
- Ollama and Qdrant are published on `127.0.0.1` only. Neither ships with authentication, so they are not exposed to the network. Keep it that way if you deploy this.
- CPU by default. To use an NVIDIA GPU, install the NVIDIA Container Toolkit and uncomment the `deploy.resources` block on the `ollama` service.
- Models and vector data live in the `ollama_models` and `qdrant_storage` volumes and survive restarts.

### Adding more characters

```bash
rag scrape --wiki marvel --character "Ben Grimm (Earth-616)"
rag scrape --wiki dc --character "Jonathan Crane (New Earth)"
rag ingest   # rebuild the index afterwards
```

You can also drop any `.md`/`.txt` files of your own into `data/corpus/`. Front matter is optional. The default character list lives in `rag/scrape.py`. Every scrape run writes `data/corpus/manifest.json` with exactly what was fetched.

## Evaluation and metrics

The eval set ([data/eval/eval_set.json](data/eval/eval_set.json)) has 33 questions. Each one is labeled with its relevant document(s), a reference answer, and a category:

| Category | Tests | Example |
|---|---|---|
| `factual` | single-document lookup | "Who is Bruce Wayne's loyal butler?" |
| `alias` | retrieval by nickname | "Which hero is known as the Man of Steel?" |
| `comparative` | multiple relevant docs, cross-universe | "Deathstroke and Deadpool are both mercenaries. What are their real names?" |
| `multi_hop` | several facts from one document | "How did Steve Rogers become Captain America, and who was his wartime partner?" |
| `unanswerable` | out-of-corpus questions | "Who is Naruto Uzumaki?" (correct behavior is to abstain) |

`rag eval` reports, overall and per category:

Retrieval (computed doc-level against the labeled relevant docs):

- Hit Rate@k: fraction of questions where at least one relevant doc was retrieved
- Precision@k / Recall@k: how much of what was retrieved is relevant / how much of the relevant set was found
- MRR: mean reciprocal rank of the first relevant doc
- nDCG@k: rank-weighted retrieval quality

Generation:

- Faithfulness (1-5): LLM-as-judge; is every claim in the answer supported by the retrieved context?
- Answer relevance (1-5): LLM-as-judge; does the answer address the question?
- Correctness (1-5): LLM-as-judge; does the answer agree with the reference answer?
- Token F1: deterministic token overlap with the reference answer (no judge involved)
- Abstention accuracy: on unanswerable questions, did the model correctly say "I don't know" instead of hallucinating?

Performance: mean/p95 end-to-end latency and mean retrieval latency.

The console shows summary tables. Complete per-question results (answers, retrieved docs, individual scores) go to `eval_results.json`.

Judge scores from a 3B model are noisy. Treat them as a signal, not ground truth. Point `RAG_JUDGE_MODEL` at a bigger model for more reliable judging.

## Configuration

Everything is overridable via `RAG_*` environment variables (see [rag/config.py](rag/config.py)):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_LLM_PROVIDER` | `ollama` | LLM backend: `ollama` or `openai` |
| `RAG_CHAT_MODEL` | `llama3.2:3b` | generation model |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `RAG_JUDGE_MODEL` | (chat model) | model used for LLM-as-judge metrics |
| `RAG_TOP_K` | `5` | chunks retrieved per query |
| `RAG_EMBED_CACHE_SIZE` | `256` | in-process LRU for query embeddings (`0` disables) |
| `RAG_FALLBACK_ENABLED` | `1` | live Fandom fetch when retrieval score is low |
| `RAG_FALLBACK_MIN_SCORE` | `0.45` | cosine threshold that triggers fallback |
| `RAG_FALLBACK_MAX_RESULTS` | `3` | MediaWiki search hits considered per wiki |
| `RAG_CHUNK_CHARS` | `1800` | max chunk size (characters) |
| `RAG_CHUNK_OVERLAP_CHARS` | `250` | overlap between adjacent chunks |
| `RAG_VECTOR_BACKEND` | `local` | vector store: `local` or `qdrant` |
| `RAG_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint |
| `RAG_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | (unset) | API key for the `openai` provider |

Example: `RAG_CHAT_MODEL=llama3.1:8b rag ask "..."`.

### Using another model provider (OpenAI-compatible)

The `openai` provider works with OpenAI itself and any server that speaks the OpenAI `/v1` API (vLLM, LM Studio, llama.cpp, LiteLLM, and similar):

```bash
# OpenAI
export RAG_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export RAG_CHAT_MODEL=gpt-4o-mini
export RAG_EMBED_MODEL=text-embedding-3-small
rag ingest && rag ask "Who trained Doctor Strange?"

# A local OpenAI-compatible server (no key needed)
export RAG_LLM_PROVIDER=openai
export RAG_OPENAI_BASE_URL=http://localhost:1234/v1
export RAG_CHAT_MODEL=your-local-model
```

Providers implement the `LLMProvider` interface (`embed`, `generate`, `available_models`, `check_ready`, `describe`) in [rag/llm.py](rag/llm.py), selected by a factory. The pipeline, ingest, and eval code depend only on that interface, so adding another provider (Gemini, Cohere, Bedrock, etc.) is a new subclass plus one line in `create_llm`. No other file changes. Switching embedding models changes vector dimensions, so re-run `rag ingest` after changing `RAG_EMBED_MODEL`.

## Project layout

```
rag/
├── config.py      # all knobs, env-overridable
├── scrape.py      # Fandom MediaWiki scraper (retries, manifest, HTML->markdown)
├── chunking.py    # markdown-section-aware chunker with overlap
├── llm.py         # pluggable LLM providers: Ollama (default) or OpenAI-compatible
├── store.py       # pluggable vector stores: local numpy (default) or Qdrant
├── ingest.py      # corpus -> chunks -> embeddings -> index
├── pipeline.py    # Retriever + RagPipeline (prompt assembly, citations, embed cache)
├── fallback.py    # low-score Fandom MediaWiki search → ingest → re-index
├── metrics.py     # retrieval metrics, token F1, LLM-as-judge scoring
├── evaluate.py    # eval runner + aggregation
└── cli.py         # typer CLI: scrape / ingest / ask / chat / eval / info
```

Each component sits behind a small interface: `LLMProvider` for embeddings/generation and `BaseVectorStore` for retrieval. Factories (`create_llm`, `create_store`/`open_store`) pick the implementation. The pipeline, ingest, and eval code depend only on those interfaces, so swapping the LLM backend or the vector store is a one-file change.

## Troubleshooting

- `Cannot reach Ollama`: start the server with `ollama serve`.
- `Missing Ollama models`: run the `ollama pull` commands from Setup.
- Everything is slow / Ollama says `100% CPU` in `ollama ps`: Ollama may be silently failing to load its GPU runner. One known cause is the runner directory not being world-readable. Check with `ls -ld /usr/local/lib/ollama/cuda_v12` and fix with `sudo chmod -R a+rX /usr/local/lib/ollama`, then restart `ollama serve`.
- Scrape failures for individual pages: page titles change on the wikis occasionally. Check `data/corpus/manifest.json` for details and pass the corrected title with `rag scrape --wiki ... --character "..."`.

## Attribution and disclaimer

This is a personal, non-commercial project built for learning and experimentation.

The corpus is fetched at runtime from the [Marvel](https://marvel.fandom.com) and [DC](https://dc.fandom.com) Fandom wikis through the [MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page). That text is authored by the Fandom communities and licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/3.0/); all credit for the content goes to those contributors. Marvel and DC characters, names, and related material are trademarks of their respective owners (Marvel Entertainment and DC Comics). This project is not affiliated with, endorsed by, or sponsored by Fandom, Marvel, or DC. It does not redistribute wiki content; the scraper downloads it locally on demand. Please use it accordingly and respect each wiki's terms of use.
