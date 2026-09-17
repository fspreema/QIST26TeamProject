"""Apply only inside the new experimental image; fail if source differs."""
from pathlib import Path

replacements = {
    "zchaff_solver.cpp": (
        "for (; new_components > 0; ++itr, --new_components)",
        "for (; new_components > 0; --new_components, (new_components > 0 ? ++itr : itr))",
    ),
    "zchaff_dbase.cpp": (
        "int displacement = _lit_pool_start - old_start;",
        "std::ptrdiff_t displacement = _lit_pool_start - old_start;",
    ),
}
for name, (old, new) in replacements.items():
    path = Path(name)
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError("Unexpected Cachet source in " + name)
    text = text.replace(old, new)
    if name == "zchaff_dbase.cpp":
        text = "#include <cstddef>\n" + text
    path.write_text(text)
