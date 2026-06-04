# Extraction Tuning Plan

This note captures the current research-backed direction for improving the
spaCy and GLiNER2 extraction surfaces without conflating them with verifier or
corpus work.

## What We Observed

- `spaCy` gives us reliable sentence boundaries and dependency structure for
  relation-style reasoning when the parser is present.
- `EntityRuler` can be used either to add deterministic spans or to augment a
  statistical `ner` component; pipeline order and `overwrite_ents` determine
  whether rule-based spans augment or replace model spans.
- `GLiNER2` supports both flat entity extraction and schema-driven structured
  extraction. The schema-first API is the right place to test richer product
  and relation output, not just single-label entity lists.
- The current two-line smoke texts only exercise a subset of the available
  spans. They should be treated as minimal smoke checks, not as coverage for
  the full extraction space.

## Manual Verification Matrix

Use the same sample texts across the major extraction surfaces so the outputs
can be compared side by side.

### Text A: Product-style entity extraction

`Apple CEO Tim Cook announced iPhone 15 in Cupertino.`

Check:
- spaCy sentence segmentation and dependency parse.
- spaCy entity spans with and without a rule overlay.
- GLiNER2 entity extraction for company, person, product, and location.
- GLiNER2 schema extraction for product metadata, if we want a richer structured
  check.

Expected focus:
- `Apple`
- `Tim Cook`
- `iPhone 15`
- `Cupertino`
- optional structured fields such as title or product metadata if the schema
  is expanded.

### Text B: Cross-sentence commitment and review ambiguity

`Alice promised to ship the report tomorrow. The team will review the draft before Friday.`

Check:
- spaCy parse labels for `Alice`, `ship`, `report`, `team`, and `review`.
- rule-based or dependency-based extraction of promise/review commitment.
- antecedent resolution for whether `the team` includes `Alice`.
- whether a model or rule set over- or under-extracts from adjacent clauses.

Expected focus:
- promise
- review commitment
- antecedent resolution
- team membership ambiguity

## Tuning Questions To Answer

1. Should `EntityRuler` precede or follow `ner` for the current extraction
   profile?
2. Do we want `overwrite_ents=True` for curated spans, or should curated spans
   only augment the statistical model?
3. Is the parser-required extraction path worth the latency, or should the
   default remain sentence segmentation plus entity extraction?
4. Does GLiNER2 perform better for our use case in flat entity mode or schema
   mode, and what label schema should become the canonical one?
5. If we later train or fine-tune, which external corpora should be used for
   evaluation and which ones should remain manual-only smoke checks?

## Next Implementation Steps

- Add richer extraction fixtures that reflect the full set of expected spans
  and relations for the two sample texts.
- Add regression tests that compare the full set of returned spans, not just a
  single expected entity.
- Compare a baseline spaCy pipeline, a parser-backed pipeline, and any
  rule-overlay variant that proves useful on the same texts.
- Compare GLiNER2 flat extraction against schema-based extraction for the same
  texts and record which shape is more stable for future tuning.
- Keep the live smoke checks opt-in, pinned, and cache-backed so downloads are
  deterministic when the tests are exercised manually.

## Source Notes

- spaCy pipeline ordering, dependency parser, and `EntityRuler` behavior.
- Sentence Transformers CrossEncoder loading, cache folder, revision pinning,
  and `predict` API.
- GLiNER2 schema-based extraction and structured output formats.
