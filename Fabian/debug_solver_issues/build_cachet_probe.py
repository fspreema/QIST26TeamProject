"""Compile diagnostic Cachet only in the separate debug folder."""
import os
import sys
import difflib
import shutil
import subprocess
from pathlib import Path

dest = Path('/debug/cachet_diagnostic')
dest.mkdir(exist_ok=True)
for path in Path('/usr/src/app/cachet').iterdir():
    if path.suffix in ('.cpp', '.h') or path.name in ('Makefile', 'README.txt'):
        shutil.copy2(path, dest / path.name)
patched = '--patched' in sys.argv
if patched:
    proposed_diff = []
    path = dest / 'zchaff_solver.cpp'
    source = path.read_text()
    old = 'for (; new_components > 0; ++itr, --new_components)'
    assert source.count(old) == 1
    before = source
    source = source.replace(old, 'for (; new_components > 0; --new_components, (new_components > 0 ? ++itr : itr))')
    proposed_diff.extend(difflib.unified_diff(before.splitlines(True), source.splitlines(True), fromfile='a/zchaff_solver.cpp', tofile='b/zchaff_solver.cpp'))
    path.write_text(source)
    path = dest / 'zchaff_dbase.cpp'
    source = path.read_text()
    old = 'int displacement = _lit_pool_start - old_start;'
    assert source.count(old) == 1
    before = source
    source = '#include <cstddef>\n' + source.replace(old, 'std::ptrdiff_t displacement = _lit_pool_start - old_start;')
    proposed_diff.extend(difflib.unified_diff(before.splitlines(True), source.splitlines(True), fromfile='a/zchaff_dbase.cpp', tofile='b/zchaff_dbase.cpp'))
    path.write_text(source)
    Path('/debug/cachet_candidate.patch').write_text(''.join(proposed_diff))
offset_trace = '--offset-trace' in sys.argv
if offset_trace:
    path = dest / 'zchaff_dbase.cpp'
    source = path.read_text()
    old = 'int displacement = _lit_pool_start - old_start;'
    assert source.count(old) == 1
    source = '#include <cstdio>\n#include <stdint.h>\n' + source.replace(old, old + '\n'
        'std::fprintf(stderr, "POINTER RELOCATION: full=%lld stored_int=%d\\n", '
        '(long long)(((intptr_t)_lit_pool_start - (intptr_t)old_start) / (intptr_t)sizeof(CLitPoolElement)), displacement);')
    path.write_text(source)
command = ['make', '-B', 'MFLAGS= ',
           'CFLAGS=-g -O1 -DNDEBUG -std=c++98 -fno-omit-frame-pointer -fsanitize=address',
           'LFLAGS=-fsanitize=address -no-pie']
release = '--release' in sys.argv
if release:
    command = ['make', '-B', 'MFLAGS= ', 'CFLAGS=-O3 -DNDEBUG -std=c++98', 'LFLAGS=-static']
result = subprocess.run(command, cwd=dest, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
Path('/debug/cachet_compile.log').write_bytes(result.stdout)
print('Compiler return code:', result.returncode, flush=True)
print(result.stdout.decode(errors='replace')[-3000:], flush=True)
if result.returncode == 0:
    os.environ['ASAN_OPTIONS'] = 'detect_leaks=0'
    output = '/debug/cachet_patched_asan_n40.json' if patched else '/debug/cachet_asan_n40.json'
    if release:
        output = '/debug/cachet_patched_release_n40.json'
    if offset_trace:
        output = '/debug/cachet_offset_trace_n40.json'
    subprocess.run(['python', '/debug/container_probe.py', '--output', output,
                    '--cwd', str(dest), '--', './cachet', '/debug/n40.cnf'], check=True)
