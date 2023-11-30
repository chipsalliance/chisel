
this_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
chisel_dir := $(realpath $(this_dir)/..)
relative_to_this_dir := $(notdir $(this_dir))

mdoc_out := $(chisel_dir)/docs/generated

buildDir ?= build
website_docs_dir = $(this_dir)/docs
install_timestamp = $(this_dir)/node_modules/timestamp

# Everything not in the website directory
# This is overly conservative but okay until we can do something more precise
chisel-non-website-from-root = \
	$(shell git -C $(chisel_dir) ls-files ':!:$(relative_to_this_dir)/*')

# The above, but absolute paths
chisel-non-website = \
	$(addprefix $(chisel_dir)/,$(chisel-non-website-from-root))

# Every file commited to git in the website directory
chisel-website-only-from-root = \
	$(shell git -C $(this_dir) ls-files)

# The above, but absolute paths
chisel-website-only = \
	$(addprefix $(this_dir)/,$(chisel-website-only-from-root))

default: build

# Using index as standin for all md files
# Note this doesn't work when a markdown file is deleted
$(mdoc_out)/index.md: $(chisel-non-website)
	cd $(chisel_dir) && sbt docs/mdoc

$(website_docs_dir):
	mkdir -p $@

$(website_docs_dir)/index.md: $(mdoc_out)/index.md | $(website_docs_dir)
	cp -R $(mdoc_out)/* $(this_dir)/docs/

generated_dir = $(this_dir)/src/pages/generated
contributors = $(generated_dir)/contributors.md
scaladoc_links = $(generated_dir)/scaladoc_links.md

$(generated_dir):
	mkdir -p $@

# Unclear what this should depend on
$(contributors): $(chisel-non-website) | $(generated_dir)
	cd $(chisel_dir) && sbt docs/determineContributors

$(scaladoc_links): $(chisel-non-website) | $(generated_dir)
	cd $(chisel_dir) && sbt docs/generateScalaDocLinks

$(install_timestamp): $(this_dir)/package.json
	cd $(this_dir) && npm install
	touch $@

install: $(install_timestamp)

build: $(this_dir)/docs/index.md $(contributors) $(scaladoc_links) $(install_timestamp) $(chisel-website-only)
	cd $(this_dir) && npm run build

# Must run build first
serve: build
	cd $(this_dir) && npm run serve

clean:
	rm -rf $(this_dir)/docs $(this_dir)/build $(this_dir)/src/pages/generated $(chisel_dir)/docs/generated

mrproper: clean
	rm -rf $(this_dir)/node_modules $(this_dir)/package-lock.json

.PHONY: clean mrproper serve install
