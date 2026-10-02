# Team inputs for the final ensemble (team_stack.py)

- `team_submission_1.tsv`: team-mate submission file, used for the first stacking pass (originally `matching_results (4).tsv`).
- `team_submission_2.tsv`: team-mate submission file, used for the second stacking pass and the French initials refinement (originally `matching_results (7).tsv`).
- `team_submission_3.tsv`: team-mate submission file from a label-free France audit (originally `matching_results (8).tsv`). Only its extra French pairs in two label-supported classes are used (see `team_stack.france_team_additions`).

All three are `matching_results.tsv`-format outputs of our team's second pipeline on the same test data. No external data is involved.
The leaderboard file (**v12i, 0.990336**) is regenerated with:

```bash
cd code/business_entity_resolution/src
ER_TEAM_FILES="../team_inputs/team_submission_1.tsv;../team_inputs/team_submission_2.tsv;../team_inputs/team_submission_3.tsv" python run_all.py
# or, after a finished run (final.py has written output/ and the blend probabilities):
python team_stack.py ../../../output/matching_results.tsv ../../../output/candidate_pairs.tsv \
    ../team_inputs/team_submission_1.tsv ../team_inputs/team_submission_2.tsv ../team_inputs/team_submission_3.tsv \
    ../../../output/matching_results.tsv ../../../output/candidate_pairs.tsv
```

The team-mate pipeline's own code should be added next to this folder, so the inputs themselves can be regenerated.
