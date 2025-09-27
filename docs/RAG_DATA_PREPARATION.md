# RAG Data Preparation Pipeline

## Purpose
This document explains how source documentation is transformed into RAG-ready artifacts for the SMS assistant. It covers two primary stages:

1. Crawling documentation with `src/crawl.py`
2. Preparing Markdown into structured JSON and vector chunks with `src/prepare.py`

The intended audience is assistant developers who need to maintain, extend, or troubleshoot the data preparation workflow.

## Prerequisites
- **`config.py` settings**: Ensure `DOC_BASE_URL`, `DATA_DIR`, `CACHE_DIR`, and chunking constants (`MIN_CHUNK_TOKENS`, `MAX_CHUNK_TOKENS`, `OVERLAP_CHUNK_TOKENS`) are set.
- **Virtual environment**: Always run scripts with the project interpreter at `/Users/sergey/Documents/Projects/assistant/venv3.13/bin/python`.
- **External services**: `src/prepare.py` depends on the unified LLM interface (`src/llm.py`) and Qdrant integration (`src/qdrant.py`). Confirm API keys and Qdrant endpoints are configured.

## Directory Layout
After running the pipeline, expect the following structure under `config.DATA_DIR` and `config.CACHE_DIR`:

```text
DATA_DIR/
  html/          # Raw HTML snapshots (mirrors site structure)
  md/            # Markdown exports enriched with Source/Crumbs/Description headers
  images/        # Downloaded image assets
  docs/          # RAG-ready JSON documents (output of prepare stage)
CACHE_DIR/
  raw/
    html/        # Cached HTTP responses for pages
    images/      # Cached image binaries and headers
```
`DATA_DIR/html/urls.json` keeps crawl metadata for incremental runs.

## Stage 1 – Crawling (`src/crawl.py`)

### Responsibilities
- **Download HTML** from the configured documentation domain while respecting `robots.txt`.
- **Normalize links** (e.g., unwrap `index.html?page=...` patterns) and ensure all links/images are absolute.
- **Convert HTML to Markdown** using `_html_to_markdown_with_headings()` with breadcrumb, description, and source metadata prepended.
- **Fetch images** referenced in each page and store them locally.
- **Persist crawl metadata** to `urls.json` for auditing and resume support.

### Key Components
- **`crawl_site()`**: Breadth-first traversal bounded by `DEFAULT_MAX_PAGES` and `DEFAULT_CRAWL_DELAY`.
- **`_fetch()`**: HTTP client with caching (`_cache_get_html()` / `_cache_set_html()`), content-type filtering, and BOM cleanup.
- **`_html_to_markdown_with_headings()`**: Cleans navigation chrome, upgrades ARIA headings, preserves breadcrumbs, and prepends metadata header lines (`Source`, `Crumbs`, `Description`).
- **`_extract_links()`**: Collects same-domain anchors and image sources used for queueing and image downloads.
- **`_save_page()` / `_save_markdown()` / `_save_image()`**: Persist HTML, Markdown, and image assets mirroring the site path.

### Outputs
- **HTML snapshots** at `DATA_DIR/html/<domain>/...`
- **Markdown documents** at `DATA_DIR/md/<domain>/...` containing the metadata header
- **Image assets** at `DATA_DIR/images/<domain>/...`
- **Crawl index** `DATA_DIR/html/urls.json` listing status codes, saved paths, titles, descriptions, outgoing links, and image references

## Stage 2 – Preparation (`src/prepare.py`)

### Responsibilities
- **Extract metadata** (`Source`, `Crumbs`, `Description`) from crawled Markdown via `process_markdown_content()`.
- **Augment images** with LLM-generated descriptions using `_augment_images_with_descriptions()` and `llm.get_image_description()`.
- **Summarize content** with `llm.get_summarization()` and merge structured results (e.g., `summary`, `keypoints`).
- **Generate smart chunks** through `smart_chunk_content()` with sentence-aware overlap and token limits.
- **Persist documents** as JSON in `DATA_DIR/docs/` and push chunk-sized payloads to Qdrant with `qdrant.store_chunk()`.

### Processing Flow
1. **File discovery**: `parse_dir()` walks `DATA_DIR/md/` and calls `process_file()` for each `.md`.
2. **Metadata extraction** (`process_markdown_content()`):
   - Regex search for header lines (case-insensitive).
   - Stores `source`, `crumbs` (list), and `description` while removing them from the body.
   - Records token counts both before and after image augmentation.
3. **Image augmentation** (`_augment_images_with_descriptions()`):
   - Detects Markdown image syntax `![alt](url)`.
   - Requests descriptions from the LLM adapter; handles raw strings or JSON responses (`description` field).
   - Inserts `*Image Description: …*` blocks immediately after each image.
4. **Summarization**:
   - `get_summarization()` output is merged; supports dicts or JSON-encoded strings wrapped in code fences.
   - Non-fatal errors are logged, allowing processing to continue without summary data.
5. **Chunk generation** (`smart_chunk_content()`):
   - `_split_into_sentences_with_images()` keeps sentences intact and ensures image descriptions stay bundled.
   - `_group_sentences_into_paragraphs()` builds paragraph lists, separating image descriptions.
   - Assembles chunks while respecting `MAX_CHUNK_TOKENS` and keeping at least `MIN_CHUNK_TOKENS`.
   - Adds overlap sentences using `_get_overlap_sentences()` and splits oversized paragraphs with `_split_large_paragraph()`.
6. **Document assembly** (`process_file()`):
   - Builds `doc_id = <filename>-<md5 hash prefix>` for change detection.
   - Writes JSON with metadata, content variants, summary fields, and chunks to `DATA_DIR/docs/<doc_id>.json`.
   - Pushes each chunk, keypoint, summary, and description to Qdrant via `store_chunk()`.

### Document JSON Schema
Each entry under `DATA_DIR/docs/` contains (non-exhaustive):

- `id`
- `source_path`
- `source`, `crumbs`, `description`
- `content` (with image descriptions) and `original_content`
- `content_tokens`, `original_content_tokens`
- Summarization outputs (e.g., `summary`, `keypoints`, `action_items` depending on prompt)
- `chunks` (list of chunk strings)

This JSON is the canonical reference for inspection and debugging.

## Data Products Overview

| Artifact | Location | Producer | Purpose |
| --- | --- | --- | --- |
| HTML snapshots | `DATA_DIR/html/` | `crawl.py` | Raw reference, change tracking |
| Markdown with headers | `DATA_DIR/md/` | `crawl.py` | Source for preparation stage |
| Image assets | `DATA_DIR/images/` | `crawl.py` | Vision-aware content enrichment |
| Crawl index | `DATA_DIR/html/urls.json` | `crawl.py` | Crawl audit trail, resume support |
| RAG document JSON | `DATA_DIR/docs/` | `prepare.py` | Canonical document bundle |
| Vector payloads | Qdrant collection | `prepare.py` | Retrieval-ready chunks, summaries, keypoints |

## Operational Guidance
- **Running the crawler**:
  ```bash
  /Users/sergey/Documents/Projects/assistant/venv3.13/bin/python src/crawl.py
  ```
  Adjust `max_pages` or `delay` in `crawl_site()` when scripting custom runs.
- **Running the preparation stage**:
  ```bash
  /Users/sergey/Documents/Projects/assistant/venv3.13/bin/python src/prepare.py
  ```
- **Idempotency**: Caching and content hashing allow repeated runs without duplicating work. Updating documentation automatically yields new `doc_id` variants due to the hash suffix.
- **Error tolerance**: Failures in image description or summarization are logged but do not stop processing. Review logs to re-run affected files if needed.
- **Extensibility**: To capture new metadata fields, extend the header patterns in `process_markdown_content()`. Adjust chunking logic by tuning the constants in `config.py` or modifying helper functions.

## Validation Checklist
- **Confirm crawler output**: Inspect `DATA_DIR/md/` and verify header lines at the top of Markdown files.
- **Check JSON documents**: Open a sample `DATA_DIR/docs/<doc_id>.json` to ensure summaries, chunks, and metadata are present.
- **Inspect Qdrant**: Use Qdrant tools or dashboards to confirm new points are ingested with the correct payload fields (`doc_id`, `chunk_id`, metadata).
- **Monitor logs**: `crawl.py` and `prepare.py` both log detailed progress, including skipped pages, summary generation, and chunk counts.

## Next Steps for Developers
- **Prompt tuning**: Adjust summarization and image description prompts in `src/llm.py` to improve downstream answer quality.
- **Chunk metadata**: Enhance `store_chunk()` to include additional context (e.g., `source`, `crumbs`) for improved retrieval filters.
- **Automation**: Integrate both scripts into scheduled jobs or CI workflows to keep the RAG index fresh as documentation evolves.
