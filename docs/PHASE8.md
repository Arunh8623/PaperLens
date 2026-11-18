# Phase 8: publication info, citation counts, full-paper links, AI Quick read

Requested 2026-11-17 (four additions). Requirement ids FR12, FR13, FR14 are new (added to `TRACEABILITY.md`).

## What you get (plain language)

1. **Published-only filter.** A checkbox "published papers only" under *Filters and extras*. Published results show a green
   **Published** badge, the venue (journal/conference, volume(issue), pages, year) and the **DOI** as a link; preprints show
   a grey **Preprint** badge. Retracted papers get a red **Retracted** badge.
2. **Citation counts.** "Cited by N (OpenAlex)" under each result and on the reader page.
3. **Full paper on click + [PDF] / [HTML].** The title opens a reader page (`/paper/<id>`) that shows the whole paper from
   arXiv in a viewer with **[PDF]** and **[HTML]** tabs and an "Open in a new tab" link. The right-hand column of every
   result has **[PDF] arxiv.org** and **[HTML]** links, like Google Scholar.
4. **Quick read (AI).** "✦ Quick read" on every result and on the reader page: Gemini reads the paper and returns *In short /
   Problem / Method / Results / Limits*. Saved on disk, so each paper is explained once.

## How to run it (Windows PowerShell; use YOUR index path and YOUR e-mail)

    # one-time: publication info + citation counts from OpenAlex (same key as the `citations` step; resumable)
    $env:OPENALEX_API_KEY = "your-openalex-key"
    python -m paperlens enrich --index data\ixs --contact YOUR_EMAIL

    # one-time per terminal: the Gemini key (free, from Google AI Studio -> "Get API key"; from memory, check the site)
    $env:GEMINI_API_KEY = "your-gemini-key"

    python -m paperlens serve-web --index data\ixs --contact YOUR_EMAIL

`enrich` writes `data\ixs.openalex_meta.jsonl` (next to the index folder); `serve-web` finds it by itself and re-reads it
whenever it grows, so you can run `enrich` again while the page is open. Quick-read summaries go to `data\ixs.quickread\`.
The start-up lines tell you whether each feature is ON. The page's first status line says how many papers have publication
data and whether Quick read is on. Without the OpenAlex file the filter still works for papers whose **authors typed a
DOI on arXiv** (shown as "DOI supplied by the authors"); without a Gemini key everything else works and Quick read says why it is off.
Options: `--meta FILE`, `--quickread-dir DIR`, `--gemini-model NAME` (or env `GEMINI_MODEL`).

## Design decisions (and why)

* **Side files, not the index.** Publication info and citations change over time and come from another source, so they are NOT
  part of the persistent index (no contract change, no re-harvest, no re-embedding, crash safety untouched). They live in
  `<index>.openalex_meta.jsonl` (format in `CONTRACTS.md` §11). A missing or half-written file simply switches the feature off.
* **OpenAlex files a published paper under the PUBLISHER's DOI.** Its docs say a published version's DOI beats a preprint's, so
  looking a paper up only by its arXiv DOI (what `citations` does) can miss exactly the published ones. `enrich` therefore
  does two passes (by arXiv DOI, then by the arXiv landing-page location) and maps results back using every place a work
  mentions arXiv. **The second pass uses a filter name read from the docs and never tried on the live API.** If OpenAlex
  rejects it, `enrich` prints a note and its summary says `location lookup: unavailable`.
  The same weakness may exist in `citations`/`graph-build` (Phase 4); check on your run: if few papers were "known to
  OpenAlex", this is a likely reason. Not changed in this phase.
* **"Published" is a written rule**: not a `preprint` AND (a journal/conference location marked published version, OR a
  non-arXiv DOI on an article/book-chapter/review); or the authors supplied a non-arXiv DOI on arXiv. A DOI supplied by authors
  can occasionally point at something other than the journal version; the badge says where the claim came from (`basis`).
* **The filter runs after ranking**: with "published only" on, the server asks the ranker for `max(5k, 50)` candidates (max 100),
  keeps the published ones in the ranker's order, and shows `k`. So it can return fewer than `k` when few of the top
  candidates are published; the status line says so. (A true pre-filter would need the data inside the index and the
  workers; not needed for this scale.)
* **Citation counts are OpenAlex's**, usually lower than Google Scholar's, and the page labels them that way.
* **Quick read runs on the server.** The key never reaches the browser and travels in a header (not the URL). Only `POST` spends
  quota and it needs a custom header (`X-PaperLens: 1`) that another website cannot send to your server, so a stray page
  cannot burn your free quota. `GET` only reads the saved copy. Two simultaneous clicks make one API call.
* **Paper text is untrusted.** The prompt tells the model not to follow instructions inside the PDF, and the page renders the
  answer as plain text (never HTML).
* **If the PDF cannot be fetched** (offline, > 15 MB, not a PDF) the summary is made from title + abstract and the page says
  "abstract only". PDF downloads are limited to one per 3 s (arXiv's rule). Gemini free tier may use submitted content to
  improve Google's products; papers are public, but do not point this at private documents.
* **The model name is a setting.** Default `gemini-3-flash-preview` (confirmed on Google's developer guide and forum, 2026-11-17:
  free tier available, no shutdown announced; preview models can be retired with ~2 weeks' notice, then the API answers 404 and
  the page tells you to pass `--gemini-model`).
* **[HTML] is a best guess.** arXiv's own paper on HTML conversion (2026): papers since 2023 get automatic HTML (about 97% produce
  some, ~75% error-free); older ones are only on the ar5iv site (~90% of papers with TeX source). We link `arxiv.org/html/<id>`
  for ids from 2312 on and `ar5iv.labs.arxiv.org/html/<id>` before; we do not check each link (that would be a request per
  result). The [PDF] link always exists.

## What was tested (verified by running, Linux sandbox)

* `tests/test_enrich.py` (7): links; published rule; ids found from locations; merge with author DOI; **a published paper filed
  under the publisher DOI is found by pass 2**; resume; rejected filter; torn side-file line.
* `tests/test_quickread.py` (9): request shape, key in header only, cache, abstract fallback, setting step-down on 400,
  clear messages for 429/404/403, no-key 503, non-JSON answer, blocked/empty answer, concurrent clicks = one call, PDF checks.
* `tests/test_web_phase8.py` (7): through the real web app and query workers: links/citations/publication info on every hit,
  **published-only returns exactly the published papers in rank order**, reader route (incl. old-style ids), POST needs the
  header, cache, no-key behaviour.
* Mutation checks (deliberately broken, tests went red, code restored): filter disabled -> 2 tests failed; pass-2 lookup
  disabled -> 2 tests failed.
* Real Chromium against the running server (fake OpenAlex data + fake Gemini): no JavaScript errors; filter, Quick read,
  title -> reader, tabs, phone width all behave; screenshots checked by eye.
* Whole suite: **507 passed** (484 before + 23 new). Also fixed an import in `tests/test_harvest_years.py` that would fail.

## NOT verified (needs your machine)

| Item | Why | How to check |
|---|---|---|
| Real OpenAlex field names / location filter | sandbox cannot reach api.openalex.org; fake follows the docs | run `enrich` on 200 papers; read the printed summary; open the side file |
| Real Gemini call (model name, `thinkingLevel`, PDF input) | sandbox cannot reach Google; fake follows the docs | click Quick read once; if it errors the message says what to change |
| Whether arXiv allows its PDF/HTML inside our viewer frame | cannot reach arXiv | open any paper; if blank use "Open in a new tab" (always works) |
| Real HTML availability of a given paper | guess by id | click [HTML] on a few |
| Windows run | built on Linux | `python -m pytest` and the commands above |

Quick check that Quick read works end to end after you set the key: `python -m paperlens serve-web ...`, search anything,
click ✦ Quick read. First summary 10-60 s; free-tier limits give "wait a minute and try again".

## Limits / next steps

* Papers that OpenAlex does not know (very new ones) show "publication status not checked"/no count; re-run `enrich` later (unknown
  papers are remembered and not asked again: delete their lines to re-ask).
* Sorting by citations is not added (not requested).
* Quick read of a paper you have not harvested is not possible (it works on indexed papers only).
