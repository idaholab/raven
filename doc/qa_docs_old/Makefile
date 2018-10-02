LATEX_FLAGS=-interaction=nonstopmode

all: requirements.pdf test_plan.pdf raven_sdd.pdf

requirements.pdf : requirements.tex ../version.tex
	pdflatex $(LATEX_FLAGS) requirements.tex

test_plan.pdf : test_plan.tex ../version.tex
	pdflatex $(LATEX_FLAGS) test_plan.tex

raven_sdd.pdf : raven_sdd.tex raven_sdd.bib ravenOverview.tex ravenStructure.tex ../version.tex ravenIntro.tex
	pdflatex $(LATEX_FLAGS) raven_sdd.tex
	bibtex raven_sdd
	pdflatex $(LATEX_FLAGS) raven_sdd.tex
	pdflatex $(LATEX_FLAGS) raven_sdd.tex


requirements.odt : requirements.tex ../version.tex
	mk4ht oolatex requirements.tex

../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../version.tex

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof *.pdf
