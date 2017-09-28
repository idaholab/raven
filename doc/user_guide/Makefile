SRCFILE = raven_user_guide
LATEX_FLAGS=-shell-escape
GUIDE_FILES = raven_user_guide.tex statisticalAnalysisExample.tex forwardSamplingExample.tex singleRunAndPlot.tex adaptiveSamplingExample.tex dataMiningExample.tex reducedOrderModelingExample.tex ravenComponents.tex ravenOverview.tex  ravenTutorial.tex introduction.tex exampleDescription.tex userGuideFormats.tex HowToRun.tex ../version.tex
MAKE_DIR = $(shell pwd)

#all: raven_user_guide.pdf
all: tidy

tidy: $(GUIDE_FILES)
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex

draft: $(GUIDE_FILES)

	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	bibtex $(SRCFILE)
	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	pdflatex $(LATEX_FLAGS) "\def\DRAFT{1}\input{$(SRCFILE).tex}"

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf

#Short method of creating version file, offical version is in make_docs.sh script
../version.tex : $(MANUAL_FILES)
	git log -1 --format="%H %an %aD" .. > ../version.tex
