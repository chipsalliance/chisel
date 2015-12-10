# Retain all intermediate files.
.SECONDARY:

SBT		?= sbt
SBT_FLAGS	?= -Dsbt.log.noformat=true

CHISEL_VERSION = $(shell "$(SBT)" $(SBT_FLAGS) "show version" | tail -n 1 | cut -d ' ' -f 2)

SRC_DIR	?= .
CHISEL_BIN ?= $(abspath $(SRC_DIR)/bin)
export CHISEL_BIN

#$(info Build Chisel $(CHISEL_VERSION))

test_src_dir := src/test/scala/ChiselTests
test_results := $(filter-out main DirChange Pads SIntOps,$(notdir $(basename $(wildcard $(test_src_dir)/*.scala))))
c_resources_dir := src/main/resources

.PHONY:	smoke publish-local check clean jenkins-build coverage scaladoc test checkstyle compile

# Define the (quick) checks we should run to validate a commit
SMOKES	?= $(addprefix chiselTests.,DirectionSpec ChiselPropSpec)
smoke:
ifneq (,$(SMOKES))
	$(SBT) $(SBT_FLAGS) "testOnly $(SMOKES)"
else
	echo "no smokes"
endif

default:	publish-local

compile:
	$(SBT) $(SBT_FLAGS) compile

publish-local:
	$(SBT) $(SBT_FLAGS) publish-local

test:
	$(SBT) $(SBT_FLAGS) test

check:	test

checkstyle:
	$(SBT) $(SBT_FLAGS) scalastyle test:scalastyle

coverage:
	$(SBT) $(SBT_FLAGS) coverage test
	$(SBT) $(SBT_FLAGS) coverageReport

clean:
	$(SBT) $(SBT_FLAGS) clean
ifneq (,$(CLEAN_DIRS))
	for dir in $(CLEAN_DIRS); do $(MAKE) -C $$dir clean; done
endif
ifneq (,$(RM_DIRS))
	$(RM) -r $(RM_DIRS)
endif

scaladoc:
	$(SBT) $(SBT_FLAGS) doc

site:
	$(SBT) $(SBT_FLAGS) make-site

# Start off clean, then run tests for all supported configurations, and publish those versions of the code.
# Then run coverage and style tests (for developer's use).
# Don't publish the coverage test code since it contains hooks/references to the coverage test package
# and we don't want code with those dependencies published.
# We need to run the coverage tests last, since Jenkins will fail the build if it can't find their results.
# We explicitly invoke Make to build the clean and test targets (rather than
#  simply list them as prerequisites) to avoid parallel make issues.
jenkins-build:
	$(MAKE) clean
	$(MAKE) test
	$(SBT) $(SBT_FLAGS) publish-local
	$(SBT) $(SBT_FLAGS) scalastyle coverage test
	$(SBT) $(SBT_FLAGS) coverageReport
