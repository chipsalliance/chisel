set -e

# Skip chisel tests if the commit message says to
# Replace ... with .. in TRAVIS_COMMIT_RANGE, see https://github.com/travis-ci/travis-ci/issues/4596
if git log --format=%B --no-merges ${TRAVIS_COMMIT_RANGE/.../..} | grep '\[skip chisel tests\]'; then
  exit 0
else
  git clone https://github.com/ucb-bar/chisel3.git
  mkdir -p chisel3/lib
  cp utils/bin/firrtl.jar chisel3/lib
  cd chisel3
  sbt ++2.11.12 "set concurrentRestrictions in Global += Tags.limit(Tags.Test, 2)" clean test
fi
