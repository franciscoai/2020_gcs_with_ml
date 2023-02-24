# Makefile to generate the documentation of raytrace
# $Id: Makefile,v 1.2 2008/11/21 20:02:32 thernis Exp $

# programs
XMLTO = xmlto
LATEX = latex

# main file
DOC = raytraceman.docbook
# XSL style sheet modification file
XSL = config.xsl


all: html

# Rules to generate html doc
html: $(DOC).xml $(XSL)
	$(XMLTO) html -m $(XSL) $(DOC).xml
	$(LATEX) tex-math-equations.tex
	dvi2bitmap --magnification=5 --scale=5 tex-math-equations.dvi

clean:
	rm -f tex-math-equations.*

