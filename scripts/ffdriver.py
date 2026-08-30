#!/usr/bin/env python3
"""Drive Firefox through geckodriver's W3C WebDriver REST API (no selenium dep).

Used to reach the AWS console through the user's existing signed-in Firefox
session (a copy of their profile, so the live profile is never mutated).
"""
import json
import subprocess
import time
import urllib.error
import urllib.request

GECKO = "/snap/bin/geckodriver"
FIREFOX = "/snap/firefox/current/usr/lib/firefox/firefox"


class Firefox:
    def __init__(self, profile: str, port: int = 4444, headless: bool = False):
        self.base = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [GECKO, "--port", str(port), "--host", "127.0.0.1"],
            stdout=open("/tmp/geckodriver.log", "wb"),
            stderr=subprocess.STDOUT,
        )
        for _ in range(40):
            time.sleep(0.5)
            try:
                urllib.request.urlopen(f"{self.base}/status", timeout=2).read()
                break
            except Exception:
                continue
        args = ["-profile", profile]
        if headless:
            args.append("-headless")
        caps = {
            "capabilities": {
                "alwaysMatch": {
                    "browserName": "firefox",
                    "moz:firefoxOptions": {"binary": FIREFOX, "args": args},
                    "pageLoadStrategy": "normal",
                    "unhandledPromptBehavior": "ignore",
                }
            }
        }
        self.session = self._post("/session", caps)["value"]["sessionId"]

    def _req(self, method: str, path: str, body=None, timeout: float = 180.0):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or "{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{path} -> {e.code}: {e.read().decode()[:600]}") from None

    def _post(self, path, body=None, timeout=180.0):
        return self._req("POST", path, body if body is not None else {}, timeout)

    def _get(self, path, timeout=180.0):
        return self._req("GET", path, None, timeout)

    def goto(self, url: str, settle: float = 4.0):
        self._post(f"/session/{self.session}/url", {"url": url})
        time.sleep(settle)

    @property
    def url(self) -> str:
        return self._get(f"/session/{self.session}/url")["value"]

    @property
    def title(self) -> str:
        return self._get(f"/session/{self.session}/title")["value"]

    def script(self, js: str, args=None):
        return self._post(
            f"/session/{self.session}/execute/sync",
            {"script": js, "args": args or []},
        )["value"]

    def text(self) -> str:
        return self.script("return document.body ? document.body.innerText : ''") or ""

    def screenshot(self, path: str):
        import base64
        png = self._get(f"/session/{self.session}/screenshot")["value"]
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(png))
        return path

    def quit(self):
        try:
            self._req("DELETE", f"/session/{self.session}")
        except Exception:
            pass
        self.proc.terminate()
