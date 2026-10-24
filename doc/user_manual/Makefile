SRCFILE = raven_user_manual
MANUAL_FILES = generated/optimizer.tex rom.tex kerasROM.tex postprocessor.tex database_data.tex OutStreamSystem.tex sampler.tex variablegroups.tex \
existing_interfaces.tex ProbabilityDistributions.tex  step.tex functions.tex ravenStructure.tex Summary.tex \
introduction.tex raven_user_manual.tex model.tex runInfo.tex libraries.tex PostProcessors/DataMining.tex \
HowToRun.tex metrics.tex \
Installation/clone.tex Installation/conda.tex Installation/linux.tex Installation/macosx.tex Installation/main.tex \
Installation/overview.tex Installation/windows.tex advanced_users_templates.tex
LATEX_FLAGS=-interaction=nonstopmode
LIB_FILES = ../../scripts/library_handler.py ../../dependencies.xml

all: raven_user_manual.pdf

raven_user_manual.pdf: $(MANUAL_FILES) ../version.tex conda_command.txt pip_commands.txt
	pdflatex $(LATEX_FLAGS) $(SRCFILE).tex || echo expected to fail
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

# The generated optimizer manual (generated/optimizer.tex) is a committed artifact regenerated from
#   the optimizer input specifications by generate_docs.sh (which imports RAVEN). Because building the
#   docs must not hard-require a built RAVEN environment (the historical catch-22), this rule attempts
#   the regeneration but falls back gracefully to the committed artifact when RAVEN cannot be imported,
#   emitting a warning instead of failing the doc build. The CI staleness gate (see
#   .github/workflows/github-actions.yml) is what actually enforces that the committed artifact is in
#   sync with the input specifications, since CI always has a built RAVEN environment available.
generated/optimizer.tex : generated/generateOptimizerDoc.py ../../ravenframework/Optimizers/*.py
	./generate_docs.sh || echo "WARNING: could not regenerate generated/optimizer.tex (RAVEN not importable?); using the committed version. Run doc/user_manual/generate_docs.sh in a built RAVEN environment and commit the result."

conda_command.txt : $(LIB_FILES)
	./create_command.sh

pip_commands.txt : $(LIB_FILES)
	./create_pip_commands.sh
