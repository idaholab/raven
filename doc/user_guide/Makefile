SRCFILE = raven_user_guide

#all: raven_user_guide.pdf
all: tidy

tidy: raven_user_guide.tex statisticalAnalysis.tex forwardSampling.tex singleRunAndPlot.tex adaptiveSampling.tex dataMining.tex reducedOrderModeling.tex ravenStructure.tex ravenOverview.tex introduction.tex manualStructure.tex manualFormats.tex HowToRun.tex

	pdflatex $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(SRCFILE).tex
	pdflatex $(SRCFILE).tex

draft: raven_user_guide.tex statisticalAnalysis.tex forwardSampling.tex singleRunAndPlot.tex adaptiveSampling.tex dataMining.tex reducedOrderModeling.tex ravenStructure.tex ravenOverview.tex introduction.tex manualStructure.tex manualFormats.tex HowToRun.tex

	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	bibtex $(SRCFILE)
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"
	pdflatex "\def\DRAFT{1}\input{$(SRCFILE).tex}"

.PHONY: clean

clean:
	@rm -f *~ *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof $(SRCFILE).pdf
