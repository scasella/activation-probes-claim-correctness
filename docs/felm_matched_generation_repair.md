# FELM Matched-Generation Annotation Repair

## Summary

The repaired pilot partially passed. Fuller reference extraction and an explicit `not_enough_evidence` label fixed the previous collapse where unsupported-by-snippet claims were mostly counted as `false`: the old distribution was `true=2`, `partially_true=7`, `false=43`; the repaired GPT-5.5 v2 distribution is `true=4`, `partially_true=12`, `false=6`, `not_enough_evidence=30` over the same 52 segments. This is a real improvement, but not a clean scaling signal: `not_enough_evidence` is 57.7% of segments, leaving only 22/52 labelable claims.

## Reference Extraction

The v2 cache fetched the same FELM pilot URL list and used trafilatura readability extraction, falling back to the original FELM `ref_contents` when extraction failed or produced shorter text.

| Metric | Value |
| --- | ---: |
| Examples | 20 |
| Reference entries | 23 |
| Unique URLs | 21 |
| Unique URLs fetched | 13 |
| Unique URL fetch failures | 8 |
| Longer-than-snippet extractions | 13 |
| Fallback / failed entries | 10 / 23 |
| Original median chars | 1,534 |
| Extracted median chars | 4,505 |
| Chosen median chars | 4,505 |

This materially improved evidence coverage, but not enough to make most regenerated Llama claims adjudicable from the references.

## Label Distribution

| Label | Old pilot | Repaired v2 |
| --- | ---: | ---: |
| `true` | 2 | 4 |
| `partially_true` | 7 | 12 |
| `false` | 43 | 6 |
| `not_enough_evidence` | n/a | 30 |

Most of the repair came from moving old spurious `false` labels into `not_enough_evidence`: 28 of the old 43 `false` rows moved there. That is exactly what the new rubric was intended to do.

## Hand Inspection

I inspected 20 labels across all categories. Eighteen looked clearly defensible under the reference-bounded rubric; two were borderline but understandable.

| Claim | v2 label | Inspection note |
| --- | --- | --- |
| 0527-02 | `true` | Mostly defensible, though it infers total capacity from a percentage. |
| 0529-00 | `true` | Defensible; cowcatcher function directly supported. |
| 0535-00 | `true` | Defensible; Stand in the Schoolhouse Door supported. |
| 0554-00 | `true` | Borderline; absence of Plato in an Olivier biography weakly supports a no-evidence claim, but `not_enough_evidence` would also be reasonable. |
| 0528-01 | `false` | Defensible; reference contradicts “lack of stealth capabilities.” |
| 0531-01 | `false` | Defensible; reference says the term “act of God” does not appear in the policy. |
| 0533-00 | `false` | Defensible; “unique situation” is contradicted by broad levirate-marriage examples. |
| 0537-00 | `false` | Defensible; reference names Linda Yaccarino, not Parag Agrawal, as Twitter CEO. |
| 0547-02 | `false` | Defensible; quoted poem text does not match reference. |
| 0550-00 | `false` | Defensible; reference gives six children, not ten. |
| 0527-00 | `partially_true` | Defensible; U.S. reactor facts partly supported, but counts/global maximum are wrong or unsupported. |
| 0528-00 | `partially_true` | Defensible; delays supported, production-run reduction not supported. |
| 0528-02 | `partially_true` | Defensible; cost concerns supported, export/recouping claim not supported. |
| 0529-01 | `partially_true` | Defensible; Europe explanation partially supported but overbroad. |
| 0530-01 | `partially_true` | Borderline; some Su-57 facts supported, NATO-name clause unsupported. |
| 0531-00 | `partially_true` | Defensible; natural-disaster coverage supported, “acts of God” phrasing and accidents overreach. |
| 0527-01 | `not_enough_evidence` | Defensible; reference lacks state-level majority plant distribution. |
| 0528-03 | `not_enough_evidence` | Defensible; viability/prospects claim not stated. |
| 0529-02 | `not_enough_evidence` | Defensible; collision avoidance and urban speed claims absent. |
| 0530-00 | `not_enough_evidence` | Defensible; reference does not discuss NATO reporting names. |

## Gate Decision

Gate outcome: **partial pass**. The repair fixed the original annotation-target bug: the judge now separates contradiction from insufficient evidence, and the inspected justifications are mostly sensible. But the `not_enough_evidence` rate is too high to treat this as a clean matched-generation FELM scaling path.

I do not recommend scaling this as the main test of the generation-mismatch hypothesis unless we precommit to analyzing only the labelable subset and accept heavy attrition. For the original question, the better next path is either a small human audit of matched-generation FELM claims or moving on with the library decision using MAUD plus the existing ChatGPT-FELM result, without waiting for this automated reference-bounded path.
