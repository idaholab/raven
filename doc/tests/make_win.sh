#!/bin/bash
# Local variable definition ----------------------------------------------------
# list of files to run.
declare -a files=(analytic_tests regression_tests_documentation)
# extension to be removed.
declare -a exts=(txt ps ds)

# Functions definition ---------------------------------------------------------
# Subroutine to remove files.
clean_files () { 
	# Remove ald the files with the selected suffixes.
	for ext in "${exts[@]}"
	do
		for file in `ls *.$ext 2> /dev/null`
		do
			rm -rf *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof raven_user_manual.pdf
		done
	done
}

# Subroutine to generate files.
gen_files () { 
        git log -1 --format="%H %an %aD" .. > ../version.tex
        python ../../developer_tools/createRegressionTestDocumentation.py
	for file in "${files[@]}"
	do
		# Generate files.
        pdflatex -interaction=nonstopmode $file.tex
        bibtex $file.tex
	pdflatex -interaction=nonstopmode $file.tex
	pdflatex -interaction=nonstopmode $file.tex

	done
}

# Clean and run ----------------------------------------------------------------
if [[ $1 = 'clean' ]]
then
    # Remove al the files with the selected extension.
    clean_files
else
    # Generate all the selected files.
    gen_files
fi
