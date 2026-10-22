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
import re
import subprocess
import sys

old, new, out = sys.argv[1:]
diff = subprocess.run(
    ['latexdiff', '--flatten', '--type=UNDERLINE', '--math-markup=whole',
     '--graphics-markup=none', old, new],
    check=True, capture_output=True, text=True,
)
text = re.sub(r'\\DIFdelFL\{images/[^}]*\}', '', diff.stdout)
items = set(re.findall(r'\\bibitem\{([^}]+)\}', text))
keep = lambda m: ', '.join(k for k in map(str.strip, m.group(1).split(',')) if k in items)
text = re.sub(r'\\cite\{([^}]+)\}', lambda m: f'\\cite{{{keep(m)}}}' if keep(m) else '', text)
open(out, 'w').write(text)
print(out)
