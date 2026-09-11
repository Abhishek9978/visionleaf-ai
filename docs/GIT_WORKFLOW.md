# Git Workflow Recommendations

These are recommendations, not enforced tooling — adapt to your own
class/team requirements.

## Branching

- `main` — always in a working, demoable state. Only merge tested code.
- `milestone/<n>-<short-name>` — one branch per milestone, e.g.
  `milestone/3-image-acquisition`. Merge to `main` via PR once the
  milestone's deliverables and tests are complete.
- `fix/<short-description>` — for bug fixes discovered after a
  milestone has already been merged.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) style:

```
feat(config): add typed settings loader with env override support
fix(logging): prevent duplicate handlers on repeated setup_logging calls
test(utils): cover case-insensitive extension validation
docs(readme): document environment variable overrides
```

Prefixes: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`.

## Suggested Milestone → Commit Flow

1. Create the milestone branch.
2. Commit in small, reviewable chunks (e.g. "feat(config): add
   config.yaml schema", then "feat(config): add settings loader",
   then "test(config): add settings tests") rather than one giant
   commit per milestone.
3. Run `pytest tests/ -v` before every commit that touches code.
4. Open a PR into `main` titled `Milestone N: <name>`, with the
   milestone's "What was completed" summary as the PR description.
5. Tag releases at meaningful milestones if useful for your grading
   timeline, e.g. `git tag milestone-1 && git push --tags`.

## What Not to Commit

Covered by `.gitignore`, but worth calling out explicitly:
- Trained model files (`assets/models/*.pkl`, `.joblib`) — these can be
  large and are regenerable; consider Git LFS or a release asset if you
  need to distribute a pretrained model.
- Log files (`logs/*.log`).
- Any real dataset used for training, if it's large or has its own
  license — document how to obtain it instead in `docs/`.
