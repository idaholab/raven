#!/bin/bash
# Local variable definition ----------------------------------------------------
# list of files to run.
declare -a files=(raven_user_guide)
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

        if [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ]  || [  "$(expr substr $(uname -s) 1 4)" == "MSYS" ]
        then
          export TEXINPUTS=.\;`cygpath -w ../tex_inputs`\;$TEXINPUTS
        else
          export TEXINPUTS=../tex_inputs/:$TEXINPUTS
        fi

        git log -1 --format="%H %an %aD" .. > ../version.tex
        python ../../scripts/TestHarness/testers/RavenUtils.py --manual-list > libraries.tex
	for file in "${files[@]}"
	do
		# Generate files.
        pdflatex -shell-escape $file.tex
        bibtex $file.tex
	pdflatex -shell-escape $file.tex
	pdflatex -shell-escape $file.tex

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
