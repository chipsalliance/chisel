buildDir ?= build
subprojects = $(buildDir)/subprojects
apis = $(buildDir)/api
docs = $(buildDir)/docs

www-docs = \
	$(shell find chisel3/docs/ -name "*.md")

www-src = \
	$(shell find docs/src/main/resources) \
	docs/src/main/tut/chisel3/docs \
	docs/src/main/tut/chisel3/index.md \
	chisel3/README.md \
	firrtl/README.md \
	chisel-testers/README.md \
	chiseltest/README.md \
	treadle/README.md \
	diagrammer/README.md

chiselTags = \
	v3.2.8 \
	v3.3.3 \
	v3.4.4

# Get the latest version of some sequence of version strings
# Usage: $(call getTags,$(foo))
define latest
$(shell echo $(1) | tr " " "\n" | sort -Vr | head -n1 | sed 's/^v//')
endef

# If NO_API is set, these variables will be supressed,
# this makes building the website *much* faster for local development
ifeq ($(NO_API),)
api-copy = $(chiselTags:v%=docs/target/site/api/%/index.html)
endif

.PHONY: all clean mrproper publish serve \
	apis-chisel3
.PRECIOUS: \
	$(subprojects)/chisel3/%/.git $(subprojects)/chisel3/%/target/scala-$(scalaVersion)/unidoc/index.html \
	$(subprojects)/firrtl/%/.git $(subprojects)/firrtl/%/target/scala-$(scalaVersion)/unidoc/index.html \
	$(subprojects)/chisel-testers/%/.git $(subprojects)/chisel-testers/%/target/scala-$(scalaVersion)/api/index.html \
	$(subprojects)/chiseltest/%/.git $(subprojects)/chiseltest/%/target/scala-$(scalaVersion)/api/index.html \
	$(subprojects)/treadle/%/.git $(subprojects)/treadle/%/target/scala-$(scalaVersion)/api/index.html \
	$(subprojects)/diagrammer/%/.git $(subprojects)/diagrammer/%/target/scala-$(scalaVersion)/api/index.html \
	$(apis)/chisel3/v%/index.html \
	docs/target/site/api/%/ docs/target/site/api/firrtl/%/ docs/target/site/api/chisel-testers/%/ \
	docs/target/site/api/chiseltest/%/ docs/target/site/api/treadle/%/ docs/target/site/api/diagrammer/%/ \
	$(apis)/%/

# Build the site into the default directory (docs/target/site)
all: docs/target/site/index.html

# Targets to build the legacy APIS of only a specific subproject
apis-chisel3: $(chiselTags:%=$(apis)/chisel3/%/index.html)

# Remove the output of all build targets
clean:
	rm -rf docs/target docs/src/main/tut/contributors.md docs/src/main/tut/chisel3/docs

# Remove everything
mrproper:
	rm -rf $(buildDir) target project/target firrtl/target treadle/target diagrammer/target

# Publish Microsite
publish: all
	sbt docs/ghpagesPushSite

# Start a Jekyll server for the site
serve: all
	(cd docs/target/site && jekyll serve)

# Build the sbt-microsite
docs/target/site/index.html: build.sbt docs/src/main/tut/contributors.md $(www-src) $(api-copy) | $(api-latest)
	sbt docs/makeMicrosite

# Determine contributors
docs/src/main/tut/contributors.md: build.sbt
	sbt contributors/determineContributors

# Build API for a subproject with a specific tag. Each build rule is
# specialized by the type of documentation to build (either
# scaladoc/"sbt doc" or unidoc/"sbt unidoc"). The version of Scala in
# use by the subproject/tag (e.g., 2.11 or 2.12) is a function of the
# tag. Consequently, the rule searches for the expected output
# directory and copies that.
$(apis)/chisel3/%/index.html: $(subprojects)/chisel3/%/.git | $(apis)/chisel3/%/
	(cd $(subprojects)/chisel3/$* && sbt unidoc)
	find $(<D) -type d -name unidoc -exec cp -r '{}'/. $(@D) ';'

# Build docs in subproject with a specific tag.
docs/src/main/tut/chisel3/docs: chisel3/.git $(www-docs)
	(cd chisel3 && sbt docs/mdoc && cp -r docs/generated ../docs/src/main/tut/chisel3/docs)

# Copy *old* API of subprojects from API directory into website
docs/target/site/api/%/index.html: $(apis)/chisel3/v%/index.html | docs/target/site/api/%/
	cp -r $(<D)/. $(@D)

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
$(subprojects)/chiseltest/%/.git:
	git clone "https://github.com/ucb-bar/chisel-testers2.git" --depth 1 --branch $* $(dir $@)
%/:
	mkdir -p $@
