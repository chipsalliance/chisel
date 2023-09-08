buildDir ?= build
subprojects = $(buildDir)/subprojects
docs = $(buildDir)/docs
website_docs_dest = docs/src/main/tut/chisel3/docs
chisel3 = ../

www-docs = \
	$(shell find $(chisel3)/docs/ -name "*.md")

www-src = \
	$(website_docs_dest) \
	$(shell find docs/src/main/resources) \
	$(shell find docs/src/main/tut) \
	$(chisel3)/README.md

# Get the latest version of some sequence of version strings
# Usage: $(call getTags,$(foo))
define latest
$(shell echo $(1) | tr " " "\n" | sort -Vr | head -n1 | sed 's/^v//')
endef


.PHONY: all clean mrproper publish serve
.PRECIOUS: \
	$(subprojects)/chisel3/%/.git $(subprojects)/chisel3/%/target/scala-$(scalaVersion)/unidoc/index.html

# Build the site into the default directory (docs/target/site)
all: docs/target/site/index.html

# Remove the output of all build targets
clean:
	rm -rf docs/target docs/src/main/tut/contributors.md $(website_docs_dest)

# Remove everything
mrproper:
	rm -rf $(buildDir) target project/target

# Publish Microsite
publish: all
	sbt docs/ghpagesPushSite

# Start a Jekyll server for the site
serve: all
	(cd docs/target/site && jekyll serve)

# Build the sbt-microsite
docs/target/site/index.html: build.sbt docs/src/main/tut/contributors.md $(www-src)
	sbt docs/makeMicrosite

# Determine contributors
docs/src/main/tut/contributors.md: build.sbt
	sbt contributors/determineContributors

# Build docs in subproject with a specific tag.
# This command is brittle as it assumes chisel3 is up one dir from here in the path to where
# we want to put the website docs.
$(website_docs_dest): $(chisel3)/docs/ $(www-docs)
	(cd $(chisel3) && sbt docs/mdoc && rm -rf website/$(website_docs_dest) && cp -r docs/generated website/$(website_docs_dest))
