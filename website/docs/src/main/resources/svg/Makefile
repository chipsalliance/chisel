buildDir = ../microsite/img

images = \
	navbar_brand.png \
	navbar_brand2x.png \
	sidebar_brand.png \
	sidebar_brand2x.png

.PHONY: all

all: $(images:%=$(buildDir)/%)

$(buildDir)/navbar_brand.png: chisel-tool.svg
	inkscape --export-area-page --export-width=44 --export-height=44 $< --export-png=$@

$(buildDir)/navbar_brand2x.png: chisel-tool.svg
	inkscape --export-area-page --export-width=88 --export-height=88 $< --export-png=$@

$(buildDir)/sidebar_brand.png: chisel-tool.svg
	inkscape --export-area-page --export-width=36 --export-height=36 $< --export-png=$@

$(buildDir)/sidebar_brand2x.png: chisel-tool.svg
	inkscape --export-area-page --export-width=72 --export-height=72 $< --export-png=$@
