#!/usr/bin/env python3
"""Attach to a long-lived geckodriver session started by ff_start.py.

geckodriver runs detached so the same Firefox window (and the user's AWS login)
survives across separate tool invocations; the session id is kept in a file.
"""
import json
import time
import urllib.error
import urllib.request

STATE = "/tmp/adl-ff-session.json"


class Session:
    def __init__(self, state_path: str = STATE):
        state = json.load(open(state_path, encoding="utf-8"))
        self.base = state["base"]
        self.session = state["session"]

    def _req(self, method, path, body=None, timeout=180.0):
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

    def goto(self, url, settle=4.0):
        self._req("POST", f"/session/{self.session}/url", {"url": url})
        time.sleep(settle)

    @property
    def url(self):
        return self._req("GET", f"/session/{self.session}/url")["value"]

    @property
    def title(self):
        return self._req("GET", f"/session/{self.session}/title")["value"]

    def script(self, js, args=None):
        return self._req("POST", f"/session/{self.session}/execute/sync",
                         {"script": js, "args": args or []})["value"]

    def text(self):
        return self.script("return document.body ? document.body.innerText : ''") or ""

    def screenshot(self, path):
        import base64
        png = self._req("GET", f"/session/{self.session}/screenshot")["value"]
        open(path, "wb").write(base64.b64decode(png))
        return path

    # --- interaction helpers (the AWS console is a dynamic React app, so we
    # --- locate controls by visible text rather than brittle CSS paths) ---

    def click_text(self, text, tags=("button", "a", "span", "label", "div"), exact=False):
        """Click the innermost clickable element whose text matches."""
        js = """
        const [needle, tags, exact] = arguments;
        const nodes = [...document.querySelectorAll(tags.join(','))];
        const hit = nodes.reverse().find(el => {
          if (el.offsetParent === null) return false;
          const t = (el.textContent || '').trim();
          return exact ? t === needle : t.includes(needle);
        });
        if (!hit) return null;
        const target = hit.closest('button,a,[role=button]') || hit;
        target.scrollIntoView({block: 'center'});
        target.click();
        return (target.textContent || '').trim().slice(0, 80);
        """
        return self.script(js, [text, list(tags), exact])

    def click_radio(self, label_contains):
        js = """
        const needle = arguments[0];
        for (const inp of document.querySelectorAll('input[type=radio]')) {
          const wrap = inp.closest('label,div,fieldset');
          const txt = wrap ? (wrap.textContent || '') : '';
          if (txt.includes(needle)) { inp.scrollIntoView({block:'center'}); inp.click(); return txt.trim().slice(0,90); }
        }
        return null;
        """
        return self.script(js, [label_contains])

    def check_all_boxes(self):
        js = """
        let n = 0;
        for (const b of document.querySelectorAll('input[type=checkbox]')) {
          if (b.offsetParent !== null && !b.checked) { b.click(); n++; }
        }
        return n;
        """
        return self.script(js)

    def inputs(self):
        js = """
        return [...document.querySelectorAll('input,textarea')]
          .filter(e => e.offsetParent !== null)
          .map(e => ({type: e.type, name: e.name || '', ph: e.placeholder || '', val: (e.value||'').slice(0,60)}));
        """
        return self.script(js)

    def buttons(self):
        js = """
        return [...document.querySelectorAll('button,a[role=button],input[type=submit]')]
          .filter(e => e.offsetParent !== null)
          .map(e => (e.textContent || e.value || '').trim())
          .filter(t => t.length && t.length < 60);
        """
        return self.script(js)

    def find(self, css):
        """Return a WebDriver element id for a CSS selector, or None."""
        try:
            return self._req("POST", f"/session/{self.session}/element",
                             {"using": "css selector", "value": css})["value"]["element-6066-11e4-a52e-4f735466cecf"]
        except RuntimeError:
            return None

    def type_into(self, css, text, clear=True):
        """Type with real key events so React-controlled inputs register."""
        el = self.find(css)
        if el is None:
            return None
        if clear:
            self._req("POST", f"/session/{self.session}/element/{el}/clear", {})
        self._req("POST", f"/session/{self.session}/element/{el}/value", {"text": text})
        return True
