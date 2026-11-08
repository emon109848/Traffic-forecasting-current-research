# Project workflow

## Scope and approvals

- Work on one approved task or milestone at a time.
- Explain significant implementation choices before editing.
- Wait for explicit user approval before starting the next major milestone.
- Inspect existing work and applicable instructions before changing files.
- Do not install dependencies, download data, or initialize Git unless authorized.
- Do not spawn sub-agents unless explicitly requested.

## Research and implementation

- Prefer faithful paper implementations; document ambiguities and deviations.
- Distinguish author-code reproduction, independent paper implementation,
  adapted implementation, and educational baseline. Do not claim official status
  for this project's implementation.
- Use primary papers and author repositories; record paper versions, source
  commits, licenses, and attribution when incorporating code.
- Never fabricate experimental results or imply unrun checks passed.
- Keep dependencies and architecture minimal; grow the scaffold as needed.
- Test important components, especially split boundaries, preprocessing leakage,
  masks, metrics, tensor shapes, and model behavior.

## Reproducibility and records

- Fit learned preprocessing on training data only.
- Record dataset provenance/version/checksums, resolved configurations, seeds,
  dependency versions, hardware, code revision, and dirty working-tree status.
- Keep datasets, checkpoints, caches, and full run outputs outside tracked Git.
- Before finishing a milestone, update docs/project_status.md and record material
  decisions in docs/decisions.md. Report checks and unresolved limitations.
- Consult docs/project_status.md at the start of a future session. Continue only
  authorized work; a proposed next task is not approval to start it.
