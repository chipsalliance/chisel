#!/usr/bin/env bash
set -e

if [ $# -ne 1 ]; then
    echo "There must be exactly one argument!"
    exit -1
fi

DUT=$1

# Run formal check only for PRs
if [ $TRAVIS_PULL_REQUEST = "false" ]; then
    echo "Not a pull request, no formal check"
    exit 0
elif git log --format=%B --no-merges $TRAVIS_BRANCH..HEAD | grep '\[skip formal checks\]'; then
    echo "Commit message says to skip formal checks"
    exit 0
else
    # $TRAVIS_BRANCH is branch targeted by PR
    # Travis does a shallow clone, checkout PR target so that we have it
    # THen return to previous branch so HEAD points to the commit we're testing
    git remote set-branches origin $TRAVIS_BRANCH && git fetch
    git checkout $TRAVIS_BRANCH
    git checkout -
    cp regress/$DUT.fir $DUT.fir
    ./scripts/formal_equiv.sh HEAD $TRAVIS_BRANCH $DUT
fi
