# Makefile for building a draft PDF (without JOSS formatting)
# Requires: pandoc (>= 3.0), xelatex (TeX Live), Menlo font (macOS)

TEX = paper-draft.tex
PDF = paper-draft.pdf
SRC = paper.md
BIB = paper.bib
LUA = joss-author.lua

PANDOC_FLAGS = -s --citeproc \
  --lua-filter=$(LUA) \
  -V monofont="Menlo" \
  -V fontsize=12pt \
  -V geometry:margin=1in

.PHONY: pdf latex clean

pdf: $(PDF)

latex: $(TEX)

$(TEX): $(SRC) $(BIB) $(LUA)
	pandoc $(SRC) $(PANDOC_FLAGS) -t latex -o $(TEX)

$(PDF): $(TEX)
	xelatex $(TEX)
	xelatex $(TEX)

clean:
	rm -f $(TEX) $(PDF) paper-draft.aux paper-draft.log paper-draft.out paper-draft.toc paper-draft.nav paper-draft.snm
