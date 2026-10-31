# Engine ownership map

MAJOR engines are E1, E2, E3. Everything marked SUPPORTING is not a graded engine.

| Engine | Kind | Owner(s) | Modules (src/paperlens/...) |
|---|---|---|---|
| E1 OS Process & Ingestion | MAJOR | Arun Hadimani, Prajwal B (confirm split) | `pipeline/` (Phase 2) |
| E1 harvester (arXiv/OpenAlex) | SUPPORTING | TO CONFIRM | `harvest/` |
| E2 Persistent Hybrid Index | MAJOR | Aditya Gavimath | `store/` |
| E3 Retrieval, keyword side | MAJOR | Kartik Hegde | `text/`, `eval/` |
| E3 Retrieval, semantic + graph side | MAJOR | TO CONFIRM | `vectors/`, `graph/` (Phase 3-4) |
| Shared contracts | shared | all four sign off | `contracts/` |
| Pre-trained embedding models | SUPPORTING | - | (we do not train models) |

Owners are taken from the project brief, which listed them two different ways.
Entries marked TO CONFIRM must be settled before Gate 0. Each owner must be able
to explain their modules alone at the review.
