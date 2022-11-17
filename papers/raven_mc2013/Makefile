#
# Makefile for LaTeX files
# Origonal Author: Craig Anslow
# Modifications by: Daniel Ballinger, Richard Martineau, Hyeongkae Park
#
# Usage:
# make quick - rapid compile and view.
# make clean - remove all temporary files.
# make print - print to printer specified in file.
# make pdf - create then view pdf file.
# make ps - create then view postscript file.
# make tech - create the postscript tech report file (requires TRCover.tex).
# make html - create the document as a web page.
# Enter document name (no suffix!)
DOC	= template
# Enter printer name
#PRINT	= enchilada

all: ${DOC}.dvi ${DOC}.ps

${DOC}.dvi: *.tex 
	latex ${DOC}.tex

${DOC}.ps: ${DOC}.dvi
	dvips -t a4 -f ${DOC}.dvi -o ${DOC}.ps

${DOC}.pdf: ${DOC}.ps
	ps2pdf13 ${DOC}.ps

print1up: ${DOC}.ps
	cat trick.ps ${DOC}.ps | lpr -P ${PRINT}

print: ${DOC}.ps
	lpr -P ${PRINT} ${DOC}.ps	

dvi: ${DOC}.dvi
	kdvi ${DOC}.dvi 

pdf: ${DOC}.pdf
#	xpdf ${DOC}.pdf 

ps: ${DOC}.ps
	gv -antialias ${DOC}.ps 

clean:
	rm -f *.aux *.bbl *.blg *.dvi *.log *.toc *~

quick: ${DOC}.dvi
	xdvi ${DOC}.dvi

# File containing LaTex to construct TechReport cover.
#TRCover.dvi: TRCover.tex
#	latex TRCover.tex

# Create the document with a tech report cover.
#tech: ${DOC}.dvi TRCover.dvi
#	dviconcat -o TR${DOC}.dvi TRCover.dvi ${DOC}.dvi
#	dvips -f TR${DOC}.dvi -o TR${DOC}.ps
#	ps2pdf TR${DOC}.ps

# need webtools, generates html file
#html: ${DOC}.tex      
#	need webtools
#	latex2html ${DOC}.tex

#printf "==================Creating DVI================\n\a"
