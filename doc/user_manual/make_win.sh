#!/bin/bash
# Local variable definition ----------------------------------------------------
# list of files to run.
declare -a files=(raven_user_manual)
# extension to be removed.
declare -a exts=(txt ps ds)

# Functions definition ---------------------------------------------------------
# Subroutine to remove files.
clean_files () {
	# Remove all the files with the selected suffixes.
	for ext in "${exts[@]}"
	do
		for file in `ls *.$ext 2> /dev/null`
		do
			rm -rf *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof *.gz raven_user_manual.pdf
		done
	done
}

# Subroutine to generate files.
gen_files () {
        git log -1 --format="%H %an %aD" .. > ../version.tex
        python ../../scripts/library_handler.py manual > libraries.tex
        bash.exe ./create_command.sh
        bash.exe ./create_pip_commands.sh
	for file in "${files[@]}"
	do
		# Generate files.
        pdflatex -interaction=nonstopmode $file.tex
        bibtex $file
	pdflatex -interaction=nonstopmode $file.tex
	pdflatex -interaction=nonstopmode $file.tex

	done
}

clean_files
gen_files
