.PHONY: clean

all: build/main.pdf

TEX=$(shell find . -name "*.tex")

build/main.pdf: $(TEX)
	latexmk -pdflatex='xelatex %O %S' -pdf main.tex -outdir=build
 
clean: 
	latexmk -pdflatex='xelatex %O %S' -pdf main.tex -outdir=build -CA
