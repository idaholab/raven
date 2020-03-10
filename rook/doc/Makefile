SRCFILE = rook
MANUAL_FILES = rook.tex
LATEX_FLAGS=-interaction=nonstopmode

all: rook_manual.pdf

rook_manual.pdf: $(MANUAL_FILES)
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf

