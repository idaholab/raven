LATEX_FLAGS=-interaction=nonstopmode

all: default raven_requirements_traceability_matrix.pdf

raven_requirements_traceability_matrix.pdf : raven_requirements_traceability_matrix.tex raven_requirements_traceability_matrix.bib intro.tex ../../version.tex ../sdd/ravenDepAndLimitations.tex
	pdflatex $(LATEX_FLAGS) raven_requirements_traceability_matrix.tex
	pdflatex $(LATEX_FLAGS) raven_requirements_traceability_matrix.tex
	pdflatex $(LATEX_FLAGS) raven_requirements_traceability_matrix.tex

../../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../../version.tex

default: *.tex
	python ../../../scripts/TestHarness/testers/RavenUtils.py --manual-list > dependencies.tex
	python ../../../developer_tools/createSQAtracebilityMatrix.py -i ../srs/requirements_list.xml -o traceability_matrix.tex 
	git log -1 --format="%H %an %aD" .. > ../../version.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof raven_requirements_traceability_matrix.pdf dependencies.tex