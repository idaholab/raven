
all: requirements.pdf test_plan.pdf

requirements.pdf : requirements.tex
	pdflatex requirements.tex

test_plan.pdf : test_plan.tex
	pdflatex test_plan.tex

requirements.odt : requirements.tex
	mk4ht oolatex requirements.tex
