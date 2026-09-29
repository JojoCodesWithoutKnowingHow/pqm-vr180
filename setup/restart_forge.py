"""Restart Forge exactly as the pod image started it, and wait until it lists the ControlNets.

Forge Neo reads ``models/ControlNet`` once, at startup, and its API has no refresh
(V.1, measured). PQM's pod installs an extension *after* Forge is up (G.5), so the
companion's ControlNet models are invisible until Forge restarts. The author's
decision at V.2: **the companion restarts Forge itself**, as an install step, so
nothing in PQM changes.

- Forge is found by ``/proc/<pid>/cmdline`` -- ``bash .../webui.sh`` and its
  ``python launch.py`` -- never by a pattern that could match this script.
- It is relaunched with **the same argv, environment and working folder** it had
  (PQM's pins and flags included), in a session of its own so it outlives this
  install step and the agent that ran it.
- Done when Forge answers on its own port *and* through the image's proxy, and its
  ControlNet list has every name given on the command line. A restart that does not
  come back fails the step, and PQM terminates the pod.

    python3 restart_forge.py noobaiInpainting noobIPAMARK1

Standard library only: it runs on the image's system Python.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request

DIRECT = os.environ.get("FORGE_DIRECT", "http://127.0.0.1:7861")
PROXIED = os.environ.get("FORGE_URL", "http://127.0.0.1:7860")
TIMEOUT_S = float(os.environ.get("FORGE_RESTART_TIMEOUT_S", "600"))
LOG = os.environ.get("FORGE_RESTART_LOG", "/workspace/logs/forge_restart.log")


def procs():
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as handle:
                argv = handle.read().split(b"\0")
        except OSError:
            continue
        yield int(pid), [a.decode(errors="replace") for a in argv if a]


def is_webui(argv):
    return len(argv) >= 2 and os.path.basename(argv[0]) == "bash" and argv[1].endswith("webui.sh")


def is_launch(argv):
    return len(argv) >= 2 and "python" in os.path.basename(argv[0]) and argv[1] == "launch.py"


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as answer:
        return answer.read().decode(errors="replace")


def listed(names):
    try:
        body = get(DIRECT + "/controlnet/model_list").lower()
        get(PROXIED + "/sdapi/v1/sd-models")
    except (OSError, ValueError):
        return False
    return all(name.lower() in body for name in names)


def main(names):
    if names and listed(names):
        print("Forge already lists %s; no restart" % ", ".join(names), flush=True)
        return 0
    webui = [(p, a) for p, a in procs() if is_webui(a)]
    if not webui:
        print("no webui.sh is running: cannot restart Forge", flush=True)
        return 3
    pid, argv = webui[0]
    with open("/proc/%d/environ" % pid, "rb") as handle:
        env = dict(x.split("=", 1) for x in handle.read().decode(errors="replace").split("\0")
                   if "=" in x)
    cwd = os.readlink("/proc/%d/cwd" % pid)
    for p, a in procs():
        if is_launch(a) or is_webui(a):
            try:
                os.kill(p, signal.SIGTERM)
            except OSError:
                pass
    for _ in range(60):
        if not any(is_launch(a) or is_webui(a) for _p, a in procs()):
            break
        time.sleep(1)
    else:
        for p, a in procs():
            if is_launch(a) or is_webui(a):
                os.kill(p, signal.SIGKILL)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "ab")
    new = subprocess.Popen(argv, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, start_new_session=True)
    print("restarted Forge as pid %d: %s" % (new.pid, " ".join(argv)), flush=True)
    t0 = time.time()
    while time.time() - t0 < TIMEOUT_S:
        if new.poll() is not None and not any(is_launch(a) for _p, a in procs()):
            print("Forge exited with %s after the restart; see %s" % (new.returncode, LOG), flush=True)
            return 4
        if listed(names):
            print("Forge back in %.0fs, listing %s" % (time.time() - t0, ", ".join(names) or "its models"),
                  flush=True)
            return 0
        time.sleep(3)
    print("Forge did not come back with %s in %.0fs; see %s" % (", ".join(names), TIMEOUT_S, LOG),
          flush=True)
    return 5


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
