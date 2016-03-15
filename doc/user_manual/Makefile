SRCFILE = raven_user_manual
MANUAL_FILES = database_data.tex OutStreamSystem.tex sampler.tex existing_interfaces.tex ProbabilityDistributions.tex  step.tex functions.tex ravenStructure.tex Summary.tex introduction.tex raven_user_manual.tex model.tex runInfo.tex install.tex version.tex

#all: raven_user_manual.pdf
all: tidy

tidy: $(MANUAL_FILES)
	pdflatex $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(SRCFILE).tex
	pdflatex $(SRCFILE).tex

draft: $(MANUAL_FILES)
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	bibtex $(SRCFILE)
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf

version.tex : version_update

.PHONY : version_update

version_update :
	git describe | sed 's/_/\\_/g'  > new_version.tex
	echo "\\\\" >> new_version.tex
	git log -1 --format="%H %an\\\\%aD" . >> new_version.tex
	if diff new_version.tex version.tex; then echo No change in version.tex; else mv new_version.tex version.tex; fi
