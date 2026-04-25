# Bootstrap Confidence Intervals

Headline uncertainty intervals use a paired bootstrap at the claim level. For each of 1000 resamples, the script samples claims with replacement and evaluates every method on the same sampled claim indices. This pairing is important for method deltas because the methods share the same underlying claims; an unpaired bootstrap would mix method uncertainty with claim-sample uncertainty and overstate variance for comparisons.

Intervals are percentile 95% confidence intervals using seed `20260424`. AUROC is recomputed on each resample; if a resample contains only one class and AUROC is undefined, that resample is skipped and counted. In the current 1000-resample headline run, no AUROC cell resamples were dropped. Brier intervals use the same sampled indices and do not have a single-class failure mode.

The judge-agreement set is bootstrapped within the 110 claims where GPT-5.4 and Kimi K2.6 assign the same correctness label, not by resampling from the full 150-claim corpus and filtering after the draw. Cohen's kappa between judges is bootstrapped over the full 150 shared claims by resampling claims and recomputing kappa on each draw.
