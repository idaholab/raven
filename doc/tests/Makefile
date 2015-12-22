SRCFILE = analytic_tests

#all: raven_user_manual.pdf
all: tidy

tidy: analytic_tests.tex attenuate.tex tensor_poly.tex

	pdflatex $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(SRCFILE).tex
	pdflatex $(SRCFILE).tex

draft: analytic_tests.tex attenuate.tex tensor_poly.tex

	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	bibtex $(SRCFILE)
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf
