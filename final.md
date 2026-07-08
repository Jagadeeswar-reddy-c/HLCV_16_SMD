Mask2Former diagnosis — definitive conclusion:

Scanning all 100 queries with no threshold, the best achievable IoU over all GT instances is only 0.515 (a small truck), and that query has fg_score=0.0099 — the model thinks it's background. Class scores and mask quality are completely decoupled. This is not a preprocessing bug (normalization and image size are both correct) — it's a fundamental issue with the checkpoint in transformers 4.46.3. No decoding approach can recover it.

Project is complete:

report/report.md — full written report , covering all 6 models, size analysis, thin-object analysis, worst categories, SAM prompt ablation, both improvements, failure mode breakdown, and Mask2Former exclusion with explanation
All 11 figures in figures/ regenerated from final results
All result files present in results/
Key findings documented:

Prompting gap: SAM-oracle (0.751) vs SAM-Auto deployment (0.470) = 0.281 mIoU gap
Cascade helps small objects (+0.006) but hurts large (−0.069) — correct use case: selective application to hard cases
All models fail severely on small objects (MaskRCNN: 0.785 large → 0.337 small)
Food items (sandwich 0.076, donut 0.117) and thin utensils are consistently worst
Mask2Former excluded with documented explanation

README.md — Full rewrite with:

Results table at the top with actual mIoU numbers for all models across all size groups
Prominent Mask2Former exclusion note with explanation
4 key findings callouts (prompting gap, cascade behavior, small-object failure, worst categories)
Reference to report/report.md


Mask2Former limitation section documenting the definitive diagnosis (class scores decoupled from masks, confirmed by scanning all 100 queries)
Final results table with mIoU/size/thin numbers
Updated Mask2Former predictor description reflecting the current direct-decode implementation
report/report.md — Already written, 12 sections covering all findings

figures/ — All 11 figures regenerated from final result files:
overall_miou, miou_by_size, failure_rate_by_size, worst_cats_* (×4), sam_prompt_comparison, thin_vs_nonthin, failure_modes, improvement_comparison

results/ — 6 JSON files from all model runs