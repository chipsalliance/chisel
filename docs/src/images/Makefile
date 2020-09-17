.PHONY: all

sources = \
	type_hierarchy.dot

all: $(sources:%.dot=%.svg) $(sources:%.dot=%.png)

%.svg: %.dot
	dot -Tsvg $< -o $@

%.png: %.dot
	dot -Tpng $< -o $@
