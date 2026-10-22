"""Diff two clean LaTeX manuscripts into a marked copy.

Usage, from the repo root:

    python scripts/mark_manuscript.py OLD.tex NEW.tex OUT.tex

This round, the old file is the first submission and the new file is the
current clean manuscript:

    python scripts/mark_manuscript.py \\
        elsarticle/first_submission/main.tex \\
        elsarticle/clean/main.tex \\
        elsarticle/marked/main.tex

Requires latexdiff. \\input files are expanded before the diff.
Added text is \\DIFadd (blue underline); removed text is \\DIFdel (red strikeout).
"""
import subprocess
import sys

old, new, out = sys.argv[1:]
diff = subprocess.run(
    ['latexdiff', '--flatten', '--type=UNDERLINE', '--math-markup=whole',
     '--graphics-markup=none', old, new],
    check=True, capture_output=True, text=True,
)
open(out, 'w').write(diff.stdout)
print(out)
