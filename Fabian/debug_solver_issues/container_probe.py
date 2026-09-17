"""Run a bounded command inside a disposable container and save its raw output."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--cwd", required=True)
parser.add_argument("--input")
parser.add_argument("--timeout", type=int, default=45)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
data = Path(args.input).read_bytes() if args.input else None
start = time.monotonic()
process = subprocess.Popen(command, cwd=args.cwd, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           start_new_session=True)
timed_out = False
try:
    out, err = process.communicate(data, timeout=args.timeout)
except subprocess.TimeoutExpired:
    timed_out = True
    os.killpg(process.pid, signal.SIGKILL)
    out, err = process.communicate()
result = dict(command=command, returncode=process.returncode, timed_out=timed_out,
              elapsed=time.monotonic()-start, stdout=out.decode(errors="replace"),
              stderr=err.decode(errors="replace"))
Path(args.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
