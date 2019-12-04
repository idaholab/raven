SRCFILE = raven_user_manual
# MANUAL_FILES = optimizer.tex rom.tex postprocessor.tex database_data.tex OutStreamSystem.tex sampler.tex existing_interfaces.tex ProbabilityDistributions.tex  step.tex functions.tex ravenStructure.tex Summary.tex introduction.tex raven_user_manual.tex model.tex runInfo.tex libraries.tex DataMining.tex HowToRun.tex metrics.tex
MANUAL_FILES = optimizer.tex rom.tex kerasROM.tex postprocessor.tex database_data.tex OutStreamSystem.tex sampler.tex variablegroups.tex \
existing_interfaces.tex ProbabilityDistributions.tex  step.tex functions.tex ravenStructure.tex Summary.tex \
introduction.tex raven_user_manual.tex model.tex runInfo.tex libraries.tex DataMining.tex HowToRun.tex metrics.tex \
Installation/clone.tex Installation/conda.tex Installation/linux.tex Installation/macosx.tex Installation/main.tex \
Installation/overview.tex Installation/windows.tex advanced_users_templates.tex
LATEX_FLAGS=-interaction=nonstopmode
LIB_FILES = ../../scripts/library_handler.py ../../dependencies.ini

all: raven_user_manual.pdf

raven_user_manual.pdf: $(MANUAL_FILES) ../version.tex conda_command.txt pip_commands.txt
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	bibtex $(SRCFILE)
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex

draft: $(MANUAL_FILES) ../version.tex conda_command.txt pip_commands.txt
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

libraries.tex : $(LIB_FILES)
	python ../../scripts/library_handler.py manual > libraries.tex

conda_command.txt : $(LIB_FILES)
	./create_command.sh

pip_commands.txt : $(LIB_FILES)
	./create_pip_commands.sh
