#!/usr/bin/env python3
"""Start a detached geckodriver + visible Firefox on the user's AWS profile."""
import json
import subprocess
import sys
import time
import urllib.request

GECKO = "/snap/bin/geckodriver"
FIREFOX = "/snap/firefox/current/usr/lib/firefox/firefox"
PROFILE = sys.argv[1] if len(sys.argv) > 1 else "/home/vivek/snap/firefox/common/.mozilla/firefox/eqnmAQXP.Profile 2"
PORT = 4455
BASE = f"http://127.0.0.1:{PORT}"

subprocess.Popen(
    [GECKO, "--port", str(PORT), "--host", "127.0.0.1"],
    stdout=open("/tmp/geckodriver.log", "wb"), stderr=subprocess.STDOUT,
    start_new_session=True,
)
for _ in range(40):
    time.sleep(0.5)
    try:
        urllib.request.urlopen(f"{BASE}/status", timeout=2).read()
        break
    except Exception:
        continue

caps = {"capabilities": {"alwaysMatch": {
    "browserName": "firefox",
    "moz:firefoxOptions": {"binary": FIREFOX, "args": ["-profile", PROFILE]},
    "pageLoadStrategy": "normal",
    "unhandledPromptBehavior": "ignore",
}}}
req = urllib.request.Request(f"{BASE}/session", data=json.dumps(caps).encode(),
                             method="POST", headers={"Content-Type": "application/json"})
sid = json.loads(urllib.request.urlopen(req, timeout=180).read())["value"]["sessionId"]
json.dump({"base": BASE, "session": sid}, open("/tmp/adl-ff-session.json", "w"))
print(json.dumps({"session": sid, "profile": PROFILE}))
