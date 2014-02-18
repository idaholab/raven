SRCFILE = raven_user_manual

all: $(SRCFILE).tex
	pdflatex $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(SRCFILE).tex
	pdflatex $(SRCFILE).tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf
