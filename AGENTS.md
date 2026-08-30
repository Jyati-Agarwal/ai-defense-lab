# AI Defense Lab contributor rules

- Read `/home/vivek/.codex/attachments/c05870ad-4a8b-4f90-8878-368f7c42d388/pasted-text-1.txt` before changing code.
- This is a shared worktree. Do not revert or overwrite another worker's files.
- Preserve the single event contract in `contracts/event.schema.json`; changes are additive only.
- No feature code may read `label`, `label.*`, attack IDs, families, or split names.
- `defend/` must never import from `sim.attacks.heldout`.
- Every displayed metric must be generated from an actual run and stored under `results/`; no placeholders.
- Synthetic PII only. Attack examples must remain truncated, watermarked, and non-operational.
- Never commit `.env`, credentials, raw model prompts, or unredacted API responses.
- Use current primary documentation. Context7 is preferred, but if its quota is exhausted, use official project docs.
- Keep changes inside the ownership boundary named in the current task.

