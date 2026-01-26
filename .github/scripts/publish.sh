#!/usr/bin/env bash

set -x

PLUGIN_VERSIONS=$(./mill show plugin.publishableVersions)
PLUGIN_MODULES=$(jq -r 'map("plugin.cross[" + . + "]")' <<< "${PLUGIN_VERSIONS}")

UNIPUBLISH_VERSIONS=$(./mill show unipublish.publishableVersions)
UNIPUBLISH_MODULES=$(jq -r 'map("unipublish[" + . + "]")' <<< "${UNIPUBLISH_VERSIONS}")

MODULES=$(jq -s '.[0] + .[1]' <(echo "${PLUGIN_MODULES}") <(echo "${UNIPUBLISH_MODULES}"))
MODULES_LIST=$(jq -r 'join(",")' <<< "${MODULES}")

VERSION=$(./mill show 'unipublish[].publishVersion' | tr -d \")
BUNDLE_NAME=$(./mill show 'unipublish[].artifactMetadata' | jq -r '.group + "." + .id + "-" + .version')

# Mill rejects --bundlename for snapshots
IS_SNAPSHOT=$(./mill show 'v.isSnapshot')
[[ "${IS_SNAPSHOT}" = "true" ]] && BUNDLE_NAME_ARG="" || BUNDLE_NAME_ARG="--bundleName ${BUNDLE_NAME}"

./mill mill.javalib.SonatypeCentralPublishModule/ \
  --shouldRelease "false" \
  ${BUNDLE_NAME_ARG} \
  --publishArtifacts "{${MODULES_LIST}}.publishArtifacts"

