# Makefile for building the JOSS paper locally
# Requires: pandoc (>= 3.0), xelatex (TeX Live), Menlo font (macOS)

TEX = paper.tex
PDF = paper.pdf
SRC = paper.md
BIB = paper.bib
LUA = joss-author.lua

PANDOC_FLAGS = -s --citeproc \
  --lua-filter=$(LUA) \
  -V monofont="Menlo" \
  -V fontsize=11pt \
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
	rm -f $(TEX) $(PDF) paper.aux paper.log paper.out paper.toc paper.nav paper.snm
