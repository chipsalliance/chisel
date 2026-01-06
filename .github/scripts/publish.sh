#!/usr/bin/env bash

set -x

# Ideally Snapshots could use mill.javalib.SonatypeCentralPublishModule/ too
# but it does not work for them on 1.0.1. It seems to publish to the wrong place.
IS_SNAPSHOT=$(./mill show unipublish.isSnapshot)
PLUGIN_VERSIONS=$(./mill show plugin.publishableVersions)
PLUGIN_MODULES=$(jq -r 'map("plugin.cross[" + . + "]")' <<< "${PLUGIN_VERSIONS}")
MODULES=$(jq -r '. + ["unipublish"]' <<< "${PLUGIN_MODULES}")
if [[ "${IS_SNAPSHOT}" = "true" ]]; then
  for mod in $(jq -r '.[]' <<< "$MODULES"); do
    ./mill ${mod}.publish
  done
else
  VERSION=$(./mill show unipublish.publishVersion | tr -d \")
  BUNDLE_NAME=$(./mill show unipublish.artifactMetadata | jq -r '.group + "." + .id + "-" + .version')
  MODULES_LIST=$(jq -r 'join(",")' <<< "${MODULES}")
  ./mill mill.javalib.SonatypeCentralPublishModule/ \
    --shouldRelease "true" \
    --bundleName "$BUNDLE_NAME" \
    --publishArtifacts "{${MODULES_LIST}}.publishArtifacts"
fi
