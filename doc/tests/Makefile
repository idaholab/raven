SRCFILE = analytic_tests
LATEX_FLAGS=-interaction=nonstopmode

#all: raven_user_manual.pdf
all: tidy

tidy: analytic_tests.tex attenuate.tex tensor_poly.tex

	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex

draft: analytic_tests.tex attenuate.tex tensor_poly.tex

	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	bibtex $(SRCFILE)
	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf
