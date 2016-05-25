SRCFILE = raven_user_manual
MANUAL_FILES = rom.tex postprocessor.tex database_data.tex OutStreamSystem.tex sampler.tex existing_interfaces.tex ProbabilityDistributions.tex  step.tex functions.tex ravenStructure.tex Summary.tex introduction.tex raven_user_manual.tex model.tex runInfo.tex install.tex version.tex libraries.tex
LATEX_FLAGS=-interaction=nonstopmode

#all: raven_user_manual.pdf
all: tidy

tidy: $(MANUAL_FILES)
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex

draft: $(MANUAL_FILES)
	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	bibtex $(SRCFILE)
	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf

version.tex : version_update

libraries.tex : library_update

.PHONY : version_update library_update

version_update :
	if git describe; then git describe | sed 's/_/\\_/g' > new_version.tex;\
	echo "\\\\" >> new_version.tex;\
	git log -1 --format="%H %an\\\\%aD" . >> new_version.tex;\
	if diff new_version.tex version.tex; then echo No change in version.tex; else mv new_version.tex version.tex; fi; fi

library_update :
	python ../../scripts/TestHarness/testers/RavenUtils.py --manual-list > new_libraries.tex
	if diff new_libraries.tex libraries.tex; then echo No change in libraries.tex; else mv new_libraries.tex libraries.tex; fi
