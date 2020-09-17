# SPDX-License-Identifier: Apache-2.0
set -e

# Use appropriate branches.
# Each stable branch of FIRRTL should have a fixed value for these branches.
CHISEL_BRANCH="master"
TREADLE_BRANCH="master"

# Skip chisel tests if the commit message says to
# Replace ... with .. in TRAVIS_COMMIT_RANGE, see https://github.com/travis-ci/travis-ci/issues/4596
if git log --format=%B --no-merges ${TRAVIS_COMMIT_RANGE/.../..} | grep '\[skip chisel tests\]'; then
  exit 0
else
  git clone https://github.com/freechipsproject/treadle.git --single-branch -b ${TREADLE_BRANCH} --depth 10
  (cd treadle && sbt $SBT_ARGS +publishLocal)
  git clone https://github.com/ucb-bar/chisel3.git --single-branch -b ${CHISEL_BRANCH}
  mkdir -p chisel3/lib
  cp utils/bin/firrtl.jar chisel3/lib
  cd chisel3
  sbt "set concurrentRestrictions in Global += Tags.limit(Tags.Test, 2)" clean test
fi
