SRCFILE = raven_user_manual

#all: raven_user_manual.pdf
all: tidy

tidy: database_data.tex OutStreamSystem.tex sampler.tex existing_interfaces.tex ProbabilityDistributions.tex  step.tex functions.tex ravenStructure.tex Summary.tex introduction.tex raven_user_manual.tex model.tex runInfo.tex install.tex

	pdflatex $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(SRCFILE).tex
	pdflatex $(SRCFILE).tex

draft: database_data.tex OutStreamSystem.tex sampler.tex existing_interfaces.tex ProbabilityDistributions.tex  step.tex functions.tex ravenStructure.tex Summary.tex introduction.tex raven_user_manual.tex model.tex runInfo.tex install.tex

	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	bibtex $(SRCFILE)
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf
