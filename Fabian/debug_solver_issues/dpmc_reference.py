"""Get bounded DPMC reference counts for the identical saved weighted formulas."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

for n in (40, 50, 60):
    start = time.monotonic()
    with open('/debug/n%d.dpmc' % n, 'rb') as source:
        planner = subprocess.Popen(['./lg/build/lg', './lg/solvers/flow-cutter-pace17/flow_cutter_pace17 -p 100'],
                                   cwd='/usr/src/app/DPMC', stdin=source, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        solver = subprocess.Popen(['./dmc/dmc', '--cf=/debug/n%d.dpmc' % n],
                                  cwd='/usr/src/app/DPMC', stdin=planner.stdout, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, start_new_session=True)
        planner.stdout.close()
        timeout = False
        try:
            out, err = solver.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(solver.pid, signal.SIGKILL)
            out, err = solver.communicate()
        finally:
            for child in (planner, solver):
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            planner.wait()
    record = dict(returncode=solver.returncode, timed_out=timeout, elapsed=time.monotonic()-start,
                  stdout=out.decode(errors='replace'), stderr=err.decode(errors='replace'))
    Path('/debug/dpmc_n%d.json' % n).write_text(json.dumps(record, indent=2))
    print(n, '\n'.join(line for line in record['stdout'].splitlines()
                       if 'exact' in line or line.startswith('c seconds')), flush=True)
