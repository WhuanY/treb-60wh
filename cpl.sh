mkdir -p output

pdflatex -output-directory=output -interaction=nonstopmode exp_table.tex

rm -f *.aux *.log *.out
