
all: requirements.pdf test_plan.pdf raven_sdd.pdf

requirements.pdf : requirements.tex
	pdflatex requirements.tex

test_plan.pdf : test_plan.tex
	pdflatex test_plan.tex

raven_sdd.pdf : raven_sdd.tex
	pdflatex raven_sdd.tex
	bibtex raven_sdd
	pdflatex raven_sdd.tex


requirements.odt : requirements.tex
	mk4ht oolatex requirements.tex
