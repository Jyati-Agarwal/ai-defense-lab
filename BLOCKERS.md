# External blockers

- The V4 evaluation is sealed and its release policy was amended after scoring. Do not invoke `defend.evaluate` against the canonical `results/` again: validate the immutable cached evidence with `scripts/release.py --verify`. The evaluator's passed-terminal restore path is deferred to a future preregistered source cycle because changing it now would invalidate the frozen defense-source hash.

## Resolved 2026-08-30 — public deployment

Both tiers are live; see `deploy/DEPLOYMENT.md`.

- Frontend: https://ai-defense-lab-lac.vercel.app (Vercel, project `mem-code/ai-defense-lab`).
- Backend: container on EC2 `<EC2_INSTANCE_ID>` in ap-south-1, proxied by Vercel rewrites.
- AWS App Runner is confirmed **unusable** for account <AWS_ACCOUNT_ID>: it stopped accepting new customers on 2026-04-30 and the account has zero existing services, so the existing-customer path does not apply. `scripts/deploy_apprunner.py` and `deploy/apprunner.json` are retained for provenance but are dead ends; `deploy/ec2.sh` is the live path.
- `APP_RUNNER_EXISTING_CUSTOMER_CONFIRMED` is therefore moot and was never set.

### Still open

- The backend has no Elastic IP, so its address changes on stop/start. `deploy/ec2.sh start` re-points the Vercel rewrite and redeploys; starting the instance by any other means leaves the frontend proxying to a dead address.
- Backend traffic between Vercel and EC2 is plain HTTP inside the proxy hop. The browser-facing path is HTTPS throughout. Fronting the host with TLS (or moving it behind Vercel-only IP allowlisting) is unfinished hardening, not a demo blocker.
- `/compose` p99 was measured at ~120 ms end-to-end from a client to Mumbai, against the brief's <100 ms target. The server-side inference itself is well under budget; the gap is network RTT plus t3.micro burst behaviour, and it has not been re-measured from a co-located client.
