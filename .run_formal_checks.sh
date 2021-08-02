#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0

set -e

if [ $# -ne 1 ]; then
    echo "There must be exactly one argument!"
    exit -1
fi

DUT=$1

# See https://docs.github.com/en/actions/reference/environment-variables
# for info about these variables

# Run formal check only for PRs, GITHUB_BASE_REF is only set for PRs
if [ ! -z "$GITHUB_BASE_REF" ]; then
    # Github Actions does a shallow clone, checkout PR target so that we have it
    # Then return to previous branch so HEAD points to the commit we're testing
    git remote set-branches origin $GITHUB_BASE_REF && git fetch
    git checkout $GITHUB_BASE_REF
    git checkout -
    cp regress/$DUT.fir $DUT.fir
    ./scripts/formal_equiv.sh HEAD $GITHUB_BASE_REF $DUT
else
    echo "Not a pull request, no formal check"
    exit 0
fi
