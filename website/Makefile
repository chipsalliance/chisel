
this_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
chisel_dir := $(realpath $(this_dir)/..)
mill_out := $(chisel_dir)/out
relative_to_this_dir := $(notdir $(this_dir))
PORT ?= 8000

mdoc_out := $(chisel_dir)/docs/generated

buildDir ?= build
website_docs_dir = $(this_dir)/docs
latest_scaladoc_dir = $(this_dir)/static/api/latest
snapshot_scaladoc_dir = $(this_dir)/static/api/snapshot
snapshot_scala3_scaladoc_dir = $(this_dir)/static/api/snapshot-scala3
snapshot_scaladoc_javadoc = $(mill_out)/unipublish/2.13/docJar.dest/javadoc
snapshot_scala3_scaladoc_javadoc = $(mill_out)/unipublish/3/docJar.dest/javadoc
generate_scala_cli = $(chisel_dir)/.github/scripts/generate-scala-cli-example.sh
latest_chisel_download = $(this_dir)/static/releases/latest/download
snapshot_chisel_download = $(this_dir)/static/releases/snapshot/download
install_timestamp = $(this_dir)/node_modules/timestamp
build_timestamp = $(this_dir)/build.timestamp

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
	cd $(chisel_dir) && ./mill mdoc

$(website_docs_dir):
	mkdir -p $@

$(website_docs_dir)/index.md: $(mdoc_out)/index.md | $(website_docs_dir)
	cp -R $(mdoc_out)/* $(this_dir)/docs/

latest_version = $(chisel_dir)/latest-version.txt

$(latest_version): $(chisel-non-website)
	cd $(chisel_dir) && ./mill show latestStableVersion | tr -d '"' > $@

$(latest_scaladoc_dir):
	mkdir -p $@

$(snapshot_scaladoc_dir):
	mkdir -p $@

$(snapshot_scala3_scaladoc_dir):
	mkdir -p $@

# Must touch target because it was built when the latest release was made
# unzip doesn't support operating on stdin so we need a temp file
$(latest_scaladoc_dir)/index.html: $(latest_version) | $(latest_scaladoc_dir)
	$(eval VER := $(shell cat $(latest_version)))
	$(eval URL := https://repo1.maven.org/maven2/org/chipsalliance/chisel_2.13/$(VER)/chisel_2.13-$(VER)-javadoc.jar)
	rm -rf $(latest_scaladoc_dir)/* && \
	  cd $(latest_scaladoc_dir) && \
	  wget -q -O temp.jar $(URL) && \
	  unzip -q temp.jar && \
	  rm -rf temp.jar
	touch $@

$(snapshot_scaladoc_javadoc)/index.html: $(chisel-non-website)
	cd $(chisel_dir) && ./mill unipublish[2.13].docJar

$(snapshot_scaladoc_dir)/index.html: $(snapshot_scaladoc_javadoc)/index.html | $(snapshot_scaladoc_dir)
	rm -rf $(snapshot_scaladoc_dir)/*
	cp -R $(snapshot_scaladoc_javadoc)/* $(snapshot_scaladoc_dir)/
	touch $@

$(snapshot_scala3_scaladoc_javadoc)/index.html: $(chisel-non-website)
	cd $(chisel_dir) && ./mill unipublish[3].docJar

$(snapshot_scala3_scaladoc_dir)/index.html: $(snapshot_scala3_scaladoc_javadoc)/index.html | $(snapshot_scala3_scaladoc_dir)
	rm -rf $(snapshot_scala3_scaladoc_dir)/*
	cp -R $(snapshot_scala3_scaladoc_javadoc)/* $(snapshot_scala3_scaladoc_dir)/
	touch $@

$(latest_chisel_download):
	mkdir -p $@

$(latest_chisel_download)/chisel-example.scala: $(latest_version) | $(latest_chisel_download)
	$(eval VER := $(shell cat $(latest_version)))
	$(generate_scala_cli) $(VER) > $@

$(snapshot_chisel_download):
	mkdir -p $@

$(snapshot_chisel_download)/chisel-example.scala: | $(snapshot_chisel_download)
	$(eval VER := $(shell cd $(chisel_dir) && ./mill show unipublish[2.13].publishVersion | tr -d \"))
	$(generate_scala_cli) $(VER) > $@

generated_dir = $(this_dir)/src/pages/generated
contributors = $(generated_dir)/contributors.md
scaladoc_links = $(generated_dir)/scaladoc_links.md

$(generated_dir):
	mkdir -p $@

# Unclear what this should depend on
$(contributors): $(chisel-non-website) | $(generated_dir)
	cd $(chisel_dir) && ./mill docs.contributorsMarkdown

$(scaladoc_links): $(chisel-non-website) | $(generated_dir)
	cd $(chisel_dir) && ./mill docs.scaladocLinks

$(install_timestamp): $(this_dir)/package.json
	cd $(this_dir) && npm ci
	touch $@

install: $(install_timestamp)

$(build_timestamp): $(this_dir)/docs/index.md $(latest_scaladoc_dir)/index.html $(snapshot_scaladoc_dir)/index.html $(snapshot_scala3_scaladoc_dir)/index.html $(contributors) $(scaladoc_links) $(install_timestamp) $(chisel-website-only) $(latest_chisel_download)/chisel-example.scala $(snapshot_chisel_download)/chisel-example.scala
	cd $(this_dir) && npm run build
	touch $@

build: $(build_timestamp)

# Use python to serve the actual build directory instead of using npm run serve
# This is needed in order for the serving of the ScalaDoc to work correctly
serve: build
	cd $(this_dir)/build && python3 -m http.server ${PORT}

clean:
	rm -rf $(this_dir)/docs \
		$(this_dir)/build \
		$(this_dir)/src/pages/generated \
		$(chisel_dir)/docs/generated \
		$(latest_chisel_download) \
		$(snapshot_chisel_download)

mrproper: clean
	rm -rf $(this_dir)/node_modules \
		$(chisel_dir)/out \
		$(latest_scaladoc_dir) \
		$(snapshot_scaladoc_dir) \
		$(snapshot_scala3_scaladoc_dir)

.PHONY: clean mrproper serve install build
