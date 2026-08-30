# Deployment (live)

Two services. The React app is served from Vercel's CDN over HTTPS; the FastAPI
backend runs as a single container on one small EC2 host. Vercel proxies the API
paths server-side, so the browser only ever talks HTTPS to one origin — there is
no mixed content and no CORS preflight in the demo path.

```
browser ──HTTPS──> Vercel (static React + rewrites) ──HTTP──> EC2 :80 ──> container :8000
```

## URLs

| Piece | Value |
| --- | --- |
| Public demo | https://ai-defense-lab-lac.vercel.app |
| Vercel project | `mem-code/ai-defense-lab` |
| Backend origin | `http://<BACKEND_HOST>` (ap-south-1, auto-assigned — changes on restart) |
| Instance | `<EC2_INSTANCE_ID>` (t3.micro, ap-south-1b) |
| Image | `<AWS_ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/ai-defense-lab:latest` |

The proxied API paths are declared in `app/frontend/vercel.json`. `VITE_API_BASE_URL`
is intentionally unset so the bundle calls same-origin paths.

## First-time setup on a fresh clone

Deployment identifiers and the live backend origin are deliberately untracked, so
a clone carries no account details:

```bash
cp deploy/deployment.local.env.example deploy/deployment.local.env
$EDITOR deploy/deployment.local.env          # account id, region, instance id
```

`app/frontend/vercel.json` is generated from `vercel.json.example` on first
`./deploy/ec2.sh start`, which substitutes the instance's current address for the
`BACKEND_HOST` placeholder.

## Operating it

```bash
./deploy/ec2.sh status    # state + public IP + /health
./deploy/ec2.sh stop      # ends compute + public-IPv4 charges
./deploy/ec2.sh start     # boots, waits for ready, re-points Vercel, redeploys
./deploy/ec2.sh destroy   # terminates the instance (prompts first)
```

`start` re-points the Vercel rewrite because the instance has no Elastic IP, so
its address changes across a stop/start. Skipping that step leaves the frontend
proxying to a dead address.

## Cost

| Item | Running 24/7 | Stopped |
| --- | --- | --- |
| t3.micro | ~$7.60/mo | $0 |
| Public IPv4 | ~$3.65/mo | $0 |
| 12 GB gp3 EBS | ~$1.15/mo | ~$1.15/mo |
| ECR (785 MB) | ~$0.08/mo | ~$0.08/mo |
| **Total** | **~$12.50/mo** | **~$1.25/mo** |

Free-tier eligible accounts may pay ~$0 for the instance and IP for the first 12
months. A `ai-defense-lab-guardrail` AWS Budget alerts `vivekxgupta0@gmail.com`
at $5/month actual and forecasted spend.

## Why this shape

- **t3.micro has 1 GB but the app peaks at ~689 MB during startup** (embeddings,
  FAISS index, SHAP) before settling to ~160 MB. Boot creates a 2 GB swapfile so
  the startup spike cannot OOM; steady state stays resident, so there is no
  swap-thrash penalty at request time.
- **No load balancer.** An ALB would add ~$18/mo — more than the rest of the
  stack. Vercel already terminates TLS in front.
- **No SSH.** The host is provisioned entirely from user-data; use
  `aws ec2 get-console-output` for boot diagnostics.
- **App Runner was ruled out**: closed to new customers as of 2026-04-30 and this
  account has no existing services, so it is ineligible. AWS points to ECS
  Express Mode; a single EC2 host is cheaper and easier to pause for a demo.

## Credentials

The deploy IAM user `<deploy-iam-user>` holds only `AmazonEC2ContainerRegistryFullAccess`
and `AmazonEC2FullAccess`. The `Billing` job-function policy was attached only to
create the budget and has been removed. The instance pulls from ECR using a
12-hour ECR token injected at boot, so there is no long-lived credential and no
instance role on the host.
