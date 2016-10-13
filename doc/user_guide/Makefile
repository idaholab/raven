SRCFILE = raven_user_guide
GUIDE_FILES = raven_user_guide.tex statisticalAnalysis.tex forwardSampling.tex singleRunAndPlot.tex adaptiveSampling.tex dataMining.tex reducedOrderModeling.tex ravenStructure.tex ravenOverview.tex introduction.tex manualStructure.tex manualFormats.tex HowToRun.tex ../version.tex

#all: raven_user_guide.pdf
all: tidy

tidy: $(GUIDE_FILES)

	pdflatex $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(SRCFILE).tex
	pdflatex $(SRCFILE).tex

draft: $(GUIDE_FILES)

	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	bibtex $(SRCFILE)
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf

#Short method of creating version file, offical version is in make_docs.sh script
../version.tex : $(MANUAL_FILES)
	git log -1 --format="%H %an %aD" .. > ../version.tex
