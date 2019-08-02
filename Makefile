buildDir ?= build
subprojects = $(buildDir)/subprojects
apis = $(buildDir)/api

scalaVersion = 2.12
scalaMinorVersion = 6

www-src = \
	$(shell find docs/src/main/tut/ -name *.md) \
	$(shell find docs/src/main/resources)
chisel-src = $(shell find chisel3/ chisel-testers/ -name *.scala)

# Get all semantic version tags for a git project in a given directory
# Usage: $(call getTags,foo)
define getTags
	$(shell cd $(1) && git tag | grep "^v\([0-9]\+\.\)\{2\}[0-9]\+$$" | head -n1)
endef

firrtlTags = $(call getTags,firrtl)
chiselTags = $(call getTags,chisel3)
testersTags = $(call getTags,chisel-testers)
treadleTags = $(call getTags,treadle)
diagrammerTags = $(call getTags,diagrammer)

api-copy = \
	docs/target/site/api/chisel3/latest/index.html \
	docs/target/site/api/firrtl/latest/index.html \
	docs/target/site/api/chisel-testers/latest/index.html \
	docs/target/site/api/treadle/latest/index.html \
	docs/target/site/api/diagrammer/latest/index.html \
	$(chiselTags:%=docs/target/site/api/chisel3/%/index.html) \
	$(firrtlTags:%=docs/target/site/api/firrtl/%/index.html) \
	$(testersTags:%=docs/target/site/api/chisel-testers/%/index.html) \
	$(treadleTags:%=docs/target/site/api/treadle/%/index.html) \
	$(diagrammerTags:%=docs/target/site/api/diagrammer/%/index.html)

.PHONY: all clean mrproper serve \
	apis-chisel apis-firrtl apis-chisel-testers apis-treadle apis-diagrammer
.PRECIOUS: \
	$(subprojects)/chisel3/%/.git $(subprojects)/chisel3/%/target/scala-$(scalaVersion)/unidoc/index.html \
	$(subprojects)/firrtl/%/.git $(subprojects)/firrtl/%/target/scala-$(scalaVersion)/unidoc/index.html \
	$(subprojects)/chisel-testers/%/.git $(subprojects)/chisel-testers/%/target/scala-$(scalaVersion)/api/index.html \
	$(subprojects)/treadle/%/.git $(subprojects)/treadle/%/target/scala-$(scalaVersion)/api/index.html \
	$(subprojects)/diagrammer/%/.git $(subprojects)/diagrammer/%/target/scala-$(scalaVersion)/api/index.html \
	$(apis)/chisel3/%/index.html $(apis)/firrtl/%/index.html $(apis)/chisel-testers/%/index.html \
	$(apis)/treadle/%/index.html $(apis)/diagrammer/%/index.html \
	docs/target/site/api/chisel3/%/ docs/target/site/api/firrtl/%/ docs/target/site/api/chisel-testers/%/ \
	docs/target/site/api/treadle/%/ docs/target/site/api/diagrammer/%/ \
	$(apis)/chisel3/%/ $(apis)/firrtl/%/ $(apis)/chisel-testers/%/ $(apis)/treadle/%/ $(apis)/diagrammer/%/

# Build the site into the default directory (docs/target/site)
all: docs/target/site/index.html

# Targets to build the legacy APIS of only a specific subproject
apis-chisel: $(chiselTags:%=$(apis)/chisel3/%/index.html)
apis-firrtl: $(firrtlTags:%=$(apis)/firrtl/%/index.html)
apis-chisel-testers: $(testersTags:%=$(apis)/chisel-testers/%/index.html)
apis-treadle: $(treadleTags:%=$(apis)/treadle/%/index.html)
apis-diagrammer: $(diagrammerTags:%=$(apis)/diagrammer/%/index.html)

# Remove the output of all build targets
clean:
	rm -rf $(buildDir)/api docs/target

# Remove everything
mrproper:
	rm -rf $(buildDir) target project/target firrtl/target treadle/target diagrammer/target

# Start a Jekyll server for the site
serve: all
	(cd docs/target/site && jekyll serve)

# Build the sbt-microsite
docs/target/site/index.html: build.sbt $(www-src) $(chisel-src) $(api-copy)
	sbt ++$(scalaVersion).$(scalaMinorVersion) docs/makeMicrosite

# Build API of subprojects
chisel3/target/scala-$(scalaVersion)/unidoc/index.html: $(shell find chisel3/src chisel-testers/src -name *.scala) | chisel3/.git
	(cd chisel3/ && sbt ++$(scalaVersion).$(scalaMinorVersion) unidoc)
firrtl/target/scala-$(scalaVersion)/unidoc/index.html: $(shell find firrtl/src -name *.scala) | firrtl/.git
	(cd firrtl/ && sbt ++$(scalaVersion).$(scalaMinorVersion) unidoc)
chisel-testers/target/scala-$(scalaVersion)/api/index.html: $(shell find chisel-testers/src -name *.scala) | chisel-testers/.git
	(cd chisel-testers/ && sbt ++$(scalaVersion).$(scalaMinorVersion) doc)
treadle/target/scala-$(scalaVersion)/api/index.html: $(shell find treadle/src -name *.scala) | treadle/.git
	(cd treadle/ && sbt ++$(scalaVersion).$(scalaMinorVersion) doc)
diagrammer/target/scala-$(scalaVersion)/api/index.html: $(shell find diagrammer/src -name *.scala) | diagrammer/.git
	(cd diagrammer/ && sbt ++$(scalaVersion).$(scalaMinorVersion) doc)

# Copy built API into site
docs/target/site/api/chisel3/latest/index.html: chisel3/target/scala-$(scalaVersion)/unidoc/index.html | docs/target/site/api/chisel3/latest/
	cp -r $(dir $<)* $(dir $@)
docs/target/site/api/firrtl/latest/index.html: firrtl/target/scala-$(scalaVersion)/unidoc/index.html | docs/target/site/api/firrtl/latest/
	cp -r $(dir $<)* $(dir $@)
docs/target/site/api/treadle/latest/index.html: treadle/target/scala-$(scalaVersion)/api/index.html | docs/target/site/api/treadle/latest/
	cp -r $(dir $<)* $(dir $@)
docs/target/site/api/chisel-testers/latest/index.html: chisel-testers/target/scala-$(scalaVersion)/api/index.html | docs/target/site/api/chisel-testers/latest/
	cp -r $(dir $<)* $(dir $@)
docs/target/site/api/diagrammer/latest/index.html: diagrammer/target/scala-$(scalaVersion)/api/index.html | docs/target/site/api/diagrammer/latest/
	cp -r $(dir $<)* $(dir $@)

# Build *old* API of subprojects
$(subprojects)/chisel3/%/target/scala-$(scalaVersion)/unidoc/index.html: | $(subprojects)/chisel3/%/.git
	(cd $(subprojects)/chisel3/$* && sbt ++$(scalaVersion).$(scalaMinorVersion) unidoc)
$(subprojects)/firrtl/%/target/scala-$(scalaVersion)/unidoc/index.html: | $(subprojects)/firrtl/%/.git
	(cd $(subprojects)/firrtl/$* && sbt ++$(scalaVersion).$(scalaMinorVersion) unidoc)
$(subprojects)/chisel-testers/%/target/scala-$(scalaVersion)/api/index.html: | $(subprojects)/chisel-testers/%/.git
	(cd $(subprojects)/chisel-testers/$* && sbt ++$(scalaVersion).$(scalaMinorVersion) doc)
$(subprojects)/treadle/%/target/scala-$(scalaVersion)/api/index.html: | $(subprojects)/treadle/%/.git
	(cd $(subprojects)/treadle/$* && sbt ++$(scalaVersion).$(scalaMinorVersion) doc)
$(subprojects)/diagrammer/%/target/scala-$(scalaVersion)/api/index.html: | $(subprojects)/diagrammer/%/.git
	(cd $(subprojects)/diagrammer/$* && sbt ++$(scalaVersion).$(scalaMinorVersion) doc)

# Copy *old* API of subprojects into API diretory
$(apis)/chisel3/%/index.html: $(subprojects)/chisel3/%/target/scala-$(scalaVersion)/unidoc/index.html | $(apis)/chisel3/%/
	cp -r $(dir $<)* $(dir $@)
$(apis)/firrtl/%/index.html: $(subprojects)/firrtl/%/target/scala-$(scalaVersion)/unidoc/index.html | $(apis)/firrtl/%/
	cp -r $(dir $<)* $(dir $@)
$(apis)/chisel-testers/%/index.html: $(subprojects)/chisel-testers/%/target/scala-$(scalaVersion)/api/index.html | $(apis)/chisel-testers/%/
	cp -r $(dir $<)* $(dir $@)
$(apis)/treadle/%/index.html: $(subprojects)/treadle/%/target/scala-$(scalaVersion)/api/index.html | $(apis)/treadle/%/
	cp -r $(dir $<)* $(dir $@)
$(apis)/diagrammer/%/index.html: $(subprojects)/diagrammer/%/target/scala-$(scalaVersion)/api/index.html | $(apis)/diagrammer/%/
	cp -r $(dir $<)* $(dir $@)

# Copy *old* API of subprojects from API directory into website
docs/target/site/api/chisel3/%/index.html: $(apis)/chisel3/%/index.html | docs/target/site/api/chisel3/%/
	cp -r $(dir $<)* $(dir $@)
docs/target/site/api/firrtl/%/index.html: $(apis)/firrtl/%/index.html | docs/target/site/api/firrtl/%/
	cp -r $(dir $<)* $(dir $@)
docs/target/site/api/chisel-testers/%/index.html: $(apis)/chisel-testers/%/index.html | docs/target/site/api/chisel-testers/%/
	cp -r $(dir $<)* $(dir $@)
docs/target/site/api/treadle/%/index.html: $(apis)/treadle/%/index.html | docs/target/site/api/treadle/%/
	cp -r $(dir $<)* $(dir $@)
docs/target/site/api/diagrammer/%/index.html: $(apis)/diagrammer/%/index.html | docs/target/site/api/diagrammer/%/
	cp -r $(dir $<)* $(dir $@)

# Utilities to either fetch submodules or create directories
%/.git:
	git submodule update --init --depth 1 $*
$(subprojects)/chisel3/%/.git:
	git clone "https://github.com/freechipsproject/chisel3.git" --depth 1 --branch $* $(dir $@)
$(subprojects)/firrtl/%/.git:
	git clone "https://github.com/freechipsproject/firrtl.git" --depth 1 --branch $* $(dir $@)
$(subprojects)/chisel-testers/%/.git:
	git clone "https://github.com/freechipsproject/chisel-testers.git" --depth 1 --branch $* $(dir $@)
$(subprojects)/treadle/%/.git:
	git clone "https://github.com/freechipsproject/treadle.git" --depth 1 --branch $* $(dir $@)
$(subprojects)/diagrammer/%/.git:
	git clone "https://github.com/freechipsproject/diagrammer.git" --depth 1 --branch $* $(dir $@)
$(apis)/chisel3/%/:
	mkdir -p $@
$(apis)/firrtl/%/:
	mkdir -p $@
$(apis)/chisel-testers/%/:
	mkdir -p $@
$(apis)/treadle/%/:
	mkdir -p $@
$(apis)/diagrammer/%/:
	mkdir -p $@
docs/target/site/api/chisel3/%/:
	mkdir -p $@
docs/target/site/api/firrtl/%/:
	mkdir -p $@
docs/target/site/api/chisel-testers/%/:
	mkdir -p $@
docs/target/site/api/treadle/%/:
	mkdir -p $@
docs/target/site/api/diagrammer/%/:
	mkdir -p $@
