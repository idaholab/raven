LATEX_FLAGS=-interaction=nonstopmode

all: raven_software_requirements_specifications_and_traceability.pdf

raven_software_requirements_specifications_and_traceability.pdf : raven_software_requirements_specifications_and_traceability.tex raven_software_requirements_specifications_and_traceability.bib intro.tex ../../version.tex ../sdd/ravenDepAndLimitations.tex
	python ../../../scripts/TestHarness/testers/RavenUtils.py --manual-list > dependencies.tex
	python ../../../developer_tools/readRequirementsAndCreateLatex.py -i ../srs/requirements_list.xml -o ../srs/requirements.tex 
	python ../../../developer_tools/createSQAtracebilityMatrix.py -i ../srs/requirements_list.xml -o ../rtr/traceability_matrix.tex 
	pdflatex $(LATEX_FLAGS) raven_software_requirements_specifications_and_traceability.tex
	pdflatex $(LATEX_FLAGS) raven_software_requirements_specifications_and_traceability.tex
	pdflatex $(LATEX_FLAGS) raven_software_requirements_specifications_and_traceability.tex

../../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../../version.tex

default: *.tex
	python ../../../scripts/TestHarness/testers/RavenUtils.py --manual-list > dependencies.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof raven_software_requirements_specifications_and_traceability.pdf dependencies.tex