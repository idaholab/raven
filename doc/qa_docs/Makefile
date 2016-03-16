
all: requirements.pdf test_plan.pdf raven_sdd.pdf

requirements.pdf : requirements.tex version.tex
	pdflatex requirements.tex

test_plan.pdf : test_plan.tex version.tex
	pdflatex test_plan.tex

raven_sdd.pdf : raven_sdd.tex raven_sdd.bib ravenOverview.tex ravenStructure.tex version.tex ravenIntro.tex
	pdflatex raven_sdd.tex
	bibtex raven_sdd
	pdflatex raven_sdd.tex
	pdflatex raven_sdd.tex


requirements.odt : requirements.tex version.tex
	mk4ht oolatex requirements.tex

version.tex : version_update

.PHONY : version_update

version_update :
	git describe | sed 's/_/\\_/g'  > new_version.tex
	echo "\\\\" >> new_version.tex
	git log -1 --format="%H %an\\\\%aD" . >> new_version.tex
	if diff new_version.tex version.tex; then echo No change in version.tex; else mv new_version.tex version.tex; fi
