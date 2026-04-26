# Defensible Claims

1. Llama 3.1 8B's structured self-reported confidence does not track claim-level correctness on MAUD merger-agreement QA, FActScore biographies, or FELM world-knowledge QA.
2. Linear probes on Llama hidden-state activations recover claim-level correctness signal that self-report does not, with AUROC around 0.77-0.80 on both MAUD and FActScore held-out test splits, and consistent paired delta around 0.26 AUROC.
3. The probe-over-self-report finding replicates across two domains that differ in label source, generation source, and content domain.
4. Raw residual features perform at least as well as Goodfire SAE features on this task, but the difference is not statistically distinguishable on current evidence.
5. A GPT-5.4 external scorer is the strongest method on MAUD; its raw AUROC is inflated by roughly 0.07 points due to same-family judge coupling, with the corrected estimate around 0.87.
6. The technique's effectiveness depends on label provenance: it works on labels that directly capture claim correctness and works less reliably on labels that capture an adjacent property such as reference support.
7. The methodology does not transfer to CUAD as prompted because claim extraction is unstable on that task.
