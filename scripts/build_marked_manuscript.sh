#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/elsarticle/revised_marked"
OLD="$ROOT/first_submission/elsarticle/main.tex"
NEW="$ROOT/elsarticle/main.tex"

mkdir -p "$OUT"
latexdiff "$OLD" "$NEW" 2>/dev/null > "$OUT/main.tex"
cp "$ROOT/elsarticle"/{dataset_characteristics.tex,results_table.tex,ablation_results.tex,bibliography.bib,flowchart.eps,feature_analysis.eps,incremental_features.png,domain_interpretation.png} "$OUT/"

python3 - "$OUT/main.tex" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path).read()
text = text.replace(
    "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}>{\\raggedright\\arraybackslash}p{2.05cm} >{\\raggedright\\arraybackslash}p{1.12cm} >{\\centering\\arraybackslash}p{1.12cm} >{\\centering\\arraybackslash}p{1.22cm} >{\\raggedright\\arraybackslash}p{4.55cm}@{}}\n\\DIFaddendFL \\toprule",
    "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}>{\\raggedright\\arraybackslash}p{2.05cm} >{\\raggedright\\arraybackslash}p{1.12cm} >{\\centering\\arraybackslash}p{1.12cm} >{\\centering\\arraybackslash}p{1.22cm} >{\\raggedright\\arraybackslash}p{4.55cm}@{}}\n\\toprule",
)
text = re.sub(
    r"\\DIFdelbeginFL %DIFDELCMD < \\end\{tabularx\}.*?\\DIFaddbeginFL \\end\{tabular\*\}\s*\\DIFaddendFL",
    lambda _: "\\end{tabular*}\n\\DIFaddendFL",
    text,
    flags=re.S,
)
text = re.sub(
    r"(\\DIFaddbeginFL \\includegraphics[^\n]+\n)\s*\\DIFaddendFL \\caption",
    lambda m: m.group(1) + "  \\caption",
    text,
)
text = re.sub(
    r"(\\label\{fig:[^}]+\})\n(\\end\{figure\})",
    lambda m: m.group(1) + "\n\\DIFaddendFL\n" + m.group(2),
    text,
)
text = text.replace(
    "for downstream model training.}\n  \\label{fig:sublime_methodology}",
    "for downstream model training.}}\n  \\label{fig:sublime_methodology}",
)
open(path, "w").write(text)
PY

cd "$OUT"
latexmk -pdf -f -interaction=nonstopmode main.tex >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null || true
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null || true
cp main.pdf "$ROOT/elsarticle/main_marked.pdf"
echo "Wrote $OUT/main.pdf and elsarticle/main_marked.pdf"
