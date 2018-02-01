# LaTeX Makefile
SRCFILE = main

all: main.pdf

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof *.spl *.nls *.ilg *.nlo $(SRCFILE).pdf

main.pdf: $(SRCFILE).tex
	pdflatex $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(SRCFILE).tex
	pdflatex $(SRCFILE).tex