# ESL letter: fixes to make at the first revision

Manuscript ID **IEEE-ESL-Nov-26-0994**, submitted 13 Nov 2026 (IEEE Author Portal), status Under Review.

Found by the long-paper audit on 13 Nov 2026, after the letter was submitted. Neither changes a number.

1. Section III, "After two initial runs that were evaluated on the test split (reported in the extended
   version), design choices were made on a separate development split": only the augmentation choice was
   made on the dev split; the undefined-name gate and the choice of deployed model were informed by
   test-split results. Say so.
2. Related Work, "the model produced only 1 syntax error in 15,328 samples, so there is nothing to learn
   from without leaking test wordings": it is 1 unique erroneous draft, and "without leaking test wordings"
   was never measured. Use: "obtained only 1 erroneous draft in 15,328 samples, too few to train on; we did
   not complete a fine-tuning run."
3. Optional: mention that the browser JavaScript port reproduces the host simulation on all 624 prompts
   (scripts/check_js_engine.sh, results/js_engine_check.json).
