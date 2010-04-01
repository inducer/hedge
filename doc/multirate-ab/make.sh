#! /bin/bash

set -e

rm -Rf out
mkdir out
python generate-mrab-diagrams.py

for i in out/*.tex; do
  i=$(basename $i)
  echo "\\input{$i}" >> out/schemelist.tex
  echo "\\newpage" >> out/schemelist.tex
done

(cd out; TEXINPUTS=..:: pdflatex mrab-scheme-list.tex)
