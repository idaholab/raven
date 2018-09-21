LATEX_FLAGS=-interaction=nonstopmode

all: raven_software_design_description.pdf

raven_software_design_description.pdf : raven_software_design_description.tex raven_software_design_description.bib ravenDesignStakeHoldersAndConcerns.tex ravenIntro.tex ../../version.tex ravenSoftwareDesing.tex dataInterfaceSystemControl.tex
	pdflatex $(LATEX_FLAGS) raven_software_design_description.tex
	bibtex raven_software_design_description
	pdflatex $(LATEX_FLAGS) raven_software_design_description.tex
	pdflatex $(LATEX_FLAGS) raven_software_design_description.tex

../version.tex : *.tex
	git log -1 --format="%H %an %aD" .. > ../version.tex

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof raven_software_design_description.pdf