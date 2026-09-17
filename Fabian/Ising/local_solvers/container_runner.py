"""Mounted read-only into a disposable container; compatible with Python 3.8."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time


def main():
    request = json.load(sys.stdin)
    timeout = request["timeout"]
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".cnf") as source:
        source.write(request["formula"])
        source.flush()
        source.seek(0)
        if request["solver"] == "tensororder":
            command = ["python", "./src/tensororder.py", "--planner=factor-Flow",
                       "--weights=mcc", "--timeout=" + str(timeout)]
            cwd = "/usr/src/app/TensorOrder"
        elif request["solver"] == "cachet":
            command = ["./cachet", source.name]
            cwd = "/usr/src/app/cachet"
        else:
            raise ValueError("Unknown solver")
        started = time.perf_counter()
        process = subprocess.Popen(command, cwd=cwd, stdin=source,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        timed_out = False
        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            out, err = process.communicate()
        finally:
            # A planner can leave descendants behind even after the solver exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(json.dumps(dict(returncode=process.returncode, timed_out=timed_out,
                              elapsed=time.perf_counter()-started,
                              stdout=out.decode(errors="replace"),
                              stderr=err.decode(errors="replace"))))


if __name__ == "__main__":
    main()
