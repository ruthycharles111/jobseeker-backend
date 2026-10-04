"""Keep the worker process alive and restart it when it exits or goes silent."""

import logging
import os
import subprocess
import sys
import threading
import time


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - supervisor - %(message)s")
log = logging.getLogger("jobseeker-supervisor")
SILENCE_LIMIT_SECONDS = 180


def run_worker() -> int:
    process = subprocess.Popen(
        [sys.executable, "-u", "/app/worker.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    last_log = {"value": time.monotonic()}

    def forward_output() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            last_log["value"] = time.monotonic()
            print(f"[worker] {line}", end="", flush=True)

    reader = threading.Thread(target=forward_output, name="worker-log-forwarder", daemon=True)
    reader.start()
    while True:
        return_code = process.poll()
        if return_code is not None:
            log.error("worker exited code=%s", return_code)
            return return_code
        if time.monotonic() - last_log["value"] > SILENCE_LIMIT_SECONDS:
            log.error("worker silent for %ss; restarting", SILENCE_LIMIT_SECONDS)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return 75
        time.sleep(30)


def main() -> None:
    restart_count = 0
    while True:
        try:
            restart_count += 1
            log.info("starting worker process restart=%s", restart_count)
            run_worker()
        except Exception:
            log.exception("supervisor recovered from monitor error")
        log.warning("restarting worker in 5 seconds")
        time.sleep(5)


if __name__ == "__main__":
    main()
