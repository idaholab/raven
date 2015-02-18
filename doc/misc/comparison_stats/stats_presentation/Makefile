%.pdf : %.svg
	inkscape --export-pdf=$@ $<

presentation.pdf : presentation.tex example.pdf example_cdf_area.pdf example_pdf_common.pdf f_z.pdf
	pdflatex presentation.tex
