
SRC     = spec.tex
SPEC    = spec.pdf

all: $(SPEC)

$(SPEC) : $(SRC)
	 pdflatex -output-format=pdf spec.tex
	 pdflatex -output-format=pdf spec.tex
	 pdflatex -output-format=pdf spec.tex

clean:
	rm -f *.aux *.log *.out *.toc
