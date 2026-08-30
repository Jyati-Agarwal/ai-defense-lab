#!/usr/bin/env bash
# Operate the single-host demo backend and keep the Vercel proxy pointed at it.
#
# The instance uses an auto-assigned public IP (an Elastic IP would add ~$3.65/mo),
# so the address changes on every stop/start. "start" therefore re-points the
# Vercel rewrite and redeploys the frontend.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV="$HERE/deployment.local.env"
if [ ! -f "$LOCAL_ENV" ]; then
  echo "missing $LOCAL_ENV — copy deployment.local.env.example and fill it in" >&2
  exit 1
fi
# shellcheck disable=SC1090
set -a; . "$LOCAL_ENV"; set +a

REGION="${AWS_REGION:?}"
INSTANCE="${EC2_INSTANCE_ID:?}"
export FRONTEND="$(cd "$HERE/.." && pwd)/app/frontend"

ip_of() {
  aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}
state_of() {
  aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE" \
    --query 'Reservations[0].Instances[0].State.Name' --output text
}

repoint() {
  local ip="$1"
  [ -f "$FRONTEND/vercel.json" ] || cp "$FRONTEND/vercel.json.example" "$FRONTEND/vercel.json"
  python3 - "$ip" <<'PY'
import json, re, sys
ip = sys.argv[1]
path = __import__("os").path.join(__import__("os").environ["FRONTEND"], "vercel.json")
cfg = json.load(open(path))
for rule in cfg["rewrites"]:
    if rule["destination"].startswith("http://"):
        rule["destination"] = re.sub(r"http://(?:[0-9.]+|BACKEND_HOST)", f"http://{ip}", rule["destination"])
json.dump(cfg, open(path, "w"), indent=2)
print(f"vercel.json now proxies to http://{ip}")
PY
}

case "${1:-status}" in
  status)
    echo "state: $(state_of)"
    ip="$(ip_of)"
    echo "ip:    $ip"
    [ "$ip" != "None" ] && curl -s -m 10 "http://$ip/health" && echo || true
    ;;
  start)
    aws ec2 start-instances --region "$REGION" --instance-ids "$INSTANCE" >/dev/null
    echo "starting; waiting for the container to report ready..."
    until [ "$(state_of)" = "running" ]; do sleep 5; done
    ip="$(ip_of)"
    until curl -sf -m 8 "http://$ip/health" | grep -q '"ready"'; do sleep 10; done
    echo "backend ready at http://$ip"
    repoint "$ip"
    (cd "$FRONTEND" && vercel deploy --prod --yes >/dev/null && echo "frontend redeployed")
    ;;
  stop)
    aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE" \
      --query 'StoppingInstances[0].CurrentState.Name' --output text
    echo "stopped: compute + public-IP charges end; only the 12GB EBS volume (~\$1.15/mo) remains"
    ;;
  destroy)
    read -r -p "Terminate $INSTANCE permanently? [y/N] " ok
    [ "$ok" = "y" ] || exit 1
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE" \
      --query 'TerminatingInstances[0].CurrentState.Name' --output text
    ;;
  *) echo "usage: $0 {status|start|stop|destroy}" >&2; exit 2 ;;
esac
