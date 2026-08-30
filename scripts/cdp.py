#!/usr/bin/env python3
"""Minimal Chrome DevTools driver over the already-running browser session.

Reuses the user's logged-in Chrome (DevToolsActivePort) so console work happens
in their real session. Chrome gates the browser websocket behind a local
approval prompt; allow_remote_debugging() answers it via GNOME RemoteDesktop.
"""
import json
import sys
import threading
import time

sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/home/vivek/.browser-use-env/lib/python3.12/site-packages")

import dbus
from websockets.sync.client import connect

PORT_FILE = "/home/vivek/.config/google-chrome/DevToolsActivePort"


def allow_remote_debugging() -> None:
    time.sleep(0.4)
    bus = dbus.SessionBus()
    manager = dbus.Interface(
        bus.get_object("org.gnome.Mutter.RemoteDesktop", "/org/gnome/Mutter/RemoteDesktop"),
        "org.gnome.Mutter.RemoteDesktop",
    )
    session = dbus.Interface(
        bus.get_object("org.gnome.Mutter.RemoteDesktop", manager.CreateSession()),
        "org.gnome.Mutter.RemoteDesktop.Session",
    )
    session.Start()
    try:
        for code in (15, 57):  # Tab, Space -> focus and press the allow button
            session.NotifyKeyboardKeycode(dbus.UInt32(code), dbus.Boolean(True))
            session.NotifyKeyboardKeycode(dbus.UInt32(code), dbus.Boolean(False))
            time.sleep(0.1)
    finally:
        session.Stop()


class Chrome:
    def __init__(self) -> None:
        port, browser_path = open(PORT_FILE, encoding="utf-8").read().splitlines()[:2]
        threading.Thread(target=allow_remote_debugging, daemon=True).start()
        self.ws = connect(f"ws://127.0.0.1:{port}{browser_path}", origin=None, max_size=None)
        self._id = 0

    def cmd(self, method: str, params: dict | None = None, session: str | None = None):
        self._id += 1
        message = {"id": self._id, "method": method, "params": params or {}}
        if session:
            message["sessionId"] = session
        self.ws.send(json.dumps(message))
        while True:
            reply = json.loads(self.ws.recv())
            if reply.get("id") == self._id:
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply.get("result", {})

    def pages(self) -> list[dict]:
        return [t for t in self.cmd("Target.getTargets")["targetInfos"] if t["type"] == "page"]

    def attach(self, target_id: str) -> str:
        return self.cmd("Target.attachToTarget", {"targetId": target_id, "flatten": True})["sessionId"]

    def find(self, needle: str) -> dict | None:
        return next((p for p in self.pages() if needle in p.get("url", "")), None)

    def new_tab(self, url: str) -> str:
        return self.cmd("Target.createTarget", {"url": url})["targetId"]

    def eval(self, session: str, expression: str, timeout: float = 30.0):
        result = self.cmd(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True, "timeout": timeout * 1000},
            session,
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"].get("text", "eval failed"))
        return result.get("result", {}).get("value")

    def goto(self, session: str, url: str, settle: float = 3.0) -> None:
        self.cmd("Page.enable", session=session)
        self.cmd("Page.navigate", {"url": url}, session)
        time.sleep(settle)
