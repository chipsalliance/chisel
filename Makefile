buildDir ?= build
subprojects = $(buildDir)/subprojects
apis = $(buildDir)/api

www-src = \
	$(shell find docs/src/main/tut/ -name "*.md") \
	$(shell find docs/src/main/resources) \
	chisel3/README.md \
	firrtl/README.md \
	chisel-testers/README.md \
	chiseltest/README.md \
	treadle/README.md \
	diagrammer/README.md

firrtlTags = \
	v1.0.0 \
	v1.0.1 \
	v1.0.2 \
	v1.1.0 \
	v1.1.1 \
	v1.1.2 \
	v1.1.3 \
	v1.1.4 \
	v1.1.5 \
	v1.1.6 \
	v1.1.7 \
	v1.2.0 \
	v1.2.1 \
	v1.2.2 \
	v1.2.3 \
	v1.2.4 \
	v1.3.0-RC1
chiselTags = \
	v3.0.0 \
	v3.0.1 \
	v3.0.2 \
	v3.1.0 \
	v3.1.1 \
	v3.1.2 \
	v3.1.3 \
	v3.1.4 \
	v3.1.5 \
	v3.1.6 \
	v3.1.7 \
	v3.1.8 \
	v3.2.0 \
	v3.2.1 \
	v3.2.2 \
	v3.2.3 \
	v3.2.4 \
	v3.3.0-RC1
testersTags = \
	v1.1.0 \
	v1.1.1 \
	v1.1.2 \
	v1.2.0 \
	v1.2.1 \
	v1.2.2 \
	v1.2.3 \
	v1.2.4 \
	v1.2.5 \
	v1.2.6 \
	v1.2.7 \
	v1.2.8 \
	v1.2.9 \
	v1.2.10 \
	v1.3.0 \
	v1.3.1 \
	v1.3.2 \
	v1.3.3 \
	v1.3.4 \
	v1.4.0-RC1
treadleTags = \
	v1.0.0 \
	v1.0.1 \
	v1.0.2 \
	v1.0.3 \
	v1.0.4 \
	v1.0.5 \
	v1.1.0 \
	v1.1.1 \
	v1.1.2 \
	v1.1.3 \
	v1.1.4 \
  v1.2.0-RC1
diagrammerTags = \
	v1.0.0 \
	v1.0.1 \
	v1.0.2 \
	v1.1.0 \
	v1.1.1 \
	v1.1.2 \
	v1.1.3 \
	v1.1.4 \
	v1.2.0-RC1
chiseltestTags = \
	v0.1.0 \
	v0.1.1 \
	v0.1.2 \
	v0.1.3 \
	v0.1.4 \
	v0.2.0-RC1

# Snapshot versions that will have their API published.
firrtlSnapshot = v1.3.0-RC1
chiselSnapshot = v3.3.0-RC1
testersSnapshot = v1.4.0-RC1
treadleSnapshot = v1.2.0-RC1
diagrammerSnapshot = v1.2.0-RC1
chiseltestSnapshot = v0.2.0-RC1

# Get the latest version of some sequence of version strings
# Usage: $(call getTags,$(foo))
define latest
$(shell echo $(1) | tr " " "\n" | sort -Vr | head -n1 | sed 's/^v//')
endef

# The "latest" version that will be pointed to by, e.g., 'api/latest'
# or 'api/firrtl/latest'.
firrtlLatest = $(call latest,$(firrtlTags))
chiselLatest = $(call latest,$(chiselTags))
testersLatest = $(call latest,$(testersTags))
treadleLatest = $(call latest,$(treadleTags))
diagrammerLatest = $(call latest,$(diagrammerTags))
chiseltestLatest = $(call latest,$(chiseltestTags))

api-latest = \
	docs/target/site/api/latest \
	docs/target/site/api/firrtl/latest \
	docs/target/site/api/chisel-testers/latest \
	docs/target/site/api/treadle/latest \
	docs/target/site/api/diagrammer/latest \
	docs/target/site/api/chiseltest/latest

api-copy = \
	$(chiselTags:v%=docs/target/site/api/%/index.html) docs/target/site/api/SNAPSHOT/index.html \
	$(firrtlTags:v%=docs/target/site/api/firrtl/%/index.html) docs/target/site/api/firrtl/SNAPSHOT/index.html \
	$(testersTags:v%=docs/target/site/api/chisel-testers/%/index.html) docs/target/site/api/chisel-testers/SNAPSHOT/index.html \
	$(chiseltestTags:v%=docs/target/site/api/chiseltest/%/index.html) docs/target/site/api/chiseltest/SNAPSHOT/index.html \
	$(treadleTags:v%=docs/target/site/api/treadle/%/index.html) docs/target/site/api/treadle/SNAPSHOT/index.html \
	$(diagrammerTags:v%=docs/target/site/api/diagrammer/%/index.html) docs/target/site/api/diagrammer/SNAPSHOT/index.html \

.PHONY: all clean mrproper publish serve \
	apis-chisel apis-firrtl apis-chisel-testers apis-treadle apis-diagrammer apis-chiseltest
.PRECIOUS: \
	$(subprojects)/chisel3/%/.git $(subprojects)/chisel3/%/target/scala-$(scalaVersion)/unidoc/index.html \
	$(subprojects)/firrtl/%/.git $(subprojects)/firrtl/%/target/scala-$(scalaVersion)/unidoc/index.html \
	$(subprojects)/chisel-testers/%/.git $(subprojects)/chisel-testers/%/target/scala-$(scalaVersion)/api/index.html \
	$(subprojects)/chiseltest/%/.git $(subprojects)/chiseltest/%/target/scala-$(scalaVersion)/api/index.html \
	$(subprojects)/treadle/%/.git $(subprojects)/treadle/%/target/scala-$(scalaVersion)/api/index.html \
	$(subprojects)/diagrammer/%/.git $(subprojects)/diagrammer/%/target/scala-$(scalaVersion)/api/index.html \
	$(apis)/chisel3/v%/index.html $(apis)/firrtl/%/index.html $(apis)/chisel-testers/%/index.html \
	$(apis)/chiseltest/%/index.html $(apis)/treadle/%/index.html $(apis)/diagrammer/%/index.html \
	docs/target/site/api/%/ docs/target/site/api/firrtl/%/ docs/target/site/api/chisel-testers/%/
	docs/target/site/api/chiseltest/%/ docs/target/site/api/treadle/%/ docs/target/site/api/diagrammer/%/ \
	$(apis)/%/

# Build the site into the default directory (docs/target/site)
all: docs/target/site/index.html

# Targets to build the legacy APIS of only a specific subproject
apis-chisel: $(chiselTags:%=$(apis)/chisel3/%/index.html) $(apis)/chisel3/$(chiselSnapshot)/index.html
apis-firrtl: $(firrtlTags:%=$(apis)/firrtl/%/index.html) $(apis)/firrtl/$(firrtlSnapshot)/index.html
apis-chisel-testers: $(testersTags:%=$(apis)/chisel-testers/%/index.html) $(apis)/chisel-testers/$(testersSnapshot)/index.html
apis-chiseltest: $(chiseltestTags:%=$(apis)/chiseltest/%/index.html)
apis-treadle: $(treadleTags:%=$(apis)/treadle/%/index.html) $(apis)/treadle/$(treadleSnapshot)/index.html
apis-diagrammer: $(diagrammerTags:%=$(apis)/diagrammer/%/index.html) $(apis)/diagrammer/$(diagrammerSnapshot)/index.html

# Remove the output of all build targets
clean:
	rm -rf docs/target docs/src/main/tut/contributors.md

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

# Copy built API into site
docs/target/site/api/latest: docs/target/site/api/$(chiselLatest)/index.html
	rm -f $@
	ln -s $(chiselLatest) $@
docs/target/site/api/firrtl/latest: docs/target/site/api/firrtl/$(firrtlLatest)/index.html
	rm -f $@
	ln -s $(firrtlLatest) $@
docs/target/site/api/treadle/latest: docs/target/site/api/treadle/$(treadleLatest)/index.html
	rm -f $@
	ln -s $(treadleLatest) $@
docs/target/site/api/chisel-testers/latest: docs/target/site/api/chisel-testers/$(testersLatest)/index.html
	rm -f $@
	ln -s $(testersLatest) $@
docs/target/site/api/diagrammer/latest: docs/target/site/api/diagrammer/$(diagrammerLatest)/index.html
	rm -f $@
	ln -s $(diagrammerLatest) $@
docs/target/site/api/chiseltest/latest: docs/target/site/api/chiseltest/$(chiseltestLatest)/index.html
	rm -f $@
	ln -s $(chiseltestLatest) $@

# Build API for a subproject with a specific tag. Each build rule is
# specialized by the type of documentation to build (either
# scaladoc/"sbt doc" or unidoc/"sbt unidoc"). The version of Scala in
# use by the subproject/tag (e.g., 2.11 or 2.12) is a function of the
# tag. Consequently, the rule searches for the expected output
# directory and copies that.
$(apis)/chisel3/%/index.html: $(subprojects)/chisel3/%/.git | $(apis)/chisel3/%/
	(cd $(subprojects)/chisel3/$* && sbt unidoc)
	find $(<D) -type d -name unidoc -exec cp -r '{}'/. $(@D) ';'
$(apis)/firrtl/%/index.html: $(subprojects)/firrtl/%/.git | $(apis)/firrtl/%/
	(cd $(subprojects)/firrtl/$* && sbt unidoc)
	find $(<D) -type d -name unidoc -exec cp -r '{}'/. $(@D) ';'
$(apis)/chisel-testers/%/index.html: $(subprojects)/chisel-testers/%/.git | $(apis)/chisel-testers/%/
	(cd $(subprojects)/chisel-testers/$* && sbt doc)
	find $(<D) -type d -name api -exec cp -r '{}'/. $(@D) ';'
$(apis)/treadle/%/index.html: $(subprojects)/treadle/%/.git | $(apis)/treadle/%/
	(cd $(subprojects)/treadle/$* && sbt doc)
	find $(<D) -type d -name api -exec cp -r '{}'/. $(@D) ';'
$(apis)/diagrammer/%/index.html: $(subprojects)/diagrammer/%/.git | $(apis)/diagrammer/%/
	(cd $(subprojects)/diagrammer/$* && sbt doc)
	find $(<D) -type d -name api -exec cp -r '{}'/. $(@D) ';'
$(apis)/chiseltest/%/index.html: $(subprojects)/chiseltest/%/.git | $(apis)/chiseltest/%/
	(cd $(subprojects)/chiseltest/$* && sbt doc)
	find $(<D) -type d -name api -exec cp -r '{}'/. $(@D) ';'

# Copy *SNAPSHOT* API of subprojects into API directory
docs/target/site/api/SNAPSHOT/index.html: $(apis)/chisel3/$(chiselSnapshot)/index.html | docs/target/site/api/SNAPSHOT/
	cp -r $(<D)/. $(@D)
docs/target/site/api/firrtl/SNAPSHOT/index.html: $(apis)/firrtl/$(firrtlSnapshot)/index.html | docs/target/site/api/firrtl/SNAPSHOT/
	cp -r $(<D)/. $(@D)
docs/target/site/api/chisel-testers/SNAPSHOT/index.html: $(apis)/chisel-testers/$(testersSnapshot)/index.html | docs/target/site/api/chisel-testers/SNAPSHOT/
	cp -r $(<D)/. $(@D)
docs/target/site/api/treadle/SNAPSHOT/index.html: $(apis)/treadle/$(treadleSnapshot)/index.html | docs/target/site/api/treadle/SNAPSHOT/
	cp -r $(<D)/. $(@D)
docs/target/site/api/diagrammer/SNAPSHOT/index.html: $(apis)/diagrammer/$(diagrammerSnapshot)/index.html | docs/target/site/api/diagrammer/SNAPSHOT/
	cp -r $(<D)/. $(@D)
docs/target/site/api/chiseltest/SNAPSHOT/index.html: $(apis)/chiseltest/$(chiseltestSnapshot)/index.html | docs/target/site/api/chiseltest/SNAPSHOT/
	cp -r $(<D)/. $(@D)

# Copy *old* API of subprojects from API directory into website
docs/target/site/api/%/index.html: $(apis)/chisel3/v%/index.html | docs/target/site/api/%/
	cp -r $(<D)/. $(@D)
docs/target/site/api/firrtl/%/index.html: $(apis)/firrtl/v%/index.html | docs/target/site/api/firrtl/%/
	cp -r $(<D)/. $(@D)
docs/target/site/api/chisel-testers/%/index.html: $(apis)/chisel-testers/v%/index.html | docs/target/site/api/chisel-testers/%/
	cp -r $(<D)/. $(@D)
docs/target/site/api/treadle/%/index.html: $(apis)/treadle/v%/index.html | docs/target/site/api/treadle/%/
	cp -r $(<D)/. $(@D)
docs/target/site/api/diagrammer/%/index.html: $(apis)/diagrammer/v%/index.html | docs/target/site/api/diagrammer/%/
	cp -r $(<D)/. $(@D)
docs/target/site/api/chiseltest/%/index.html: $(apis)/chiseltest/v%/index.html | docs/target/site/api/chiseltest/%/
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
