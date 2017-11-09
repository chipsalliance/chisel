# Retain all intermediate files.
.SECONDARY:

SBT		?= sbt
SBT_FLAGS	?= -Dsbt.log.noformat=true

CHISEL_VERSION = $(shell "$(SBT)" $(SBT_FLAGS) "show version" | tail -n 1 | cut -d ' ' -f 2)

SRC_DIR	?= .
CHISEL_BIN ?= $(abspath $(SRC_DIR)/bin)
export CHISEL_BIN

#$(info Build Chisel $(CHISEL_VERSION))

# The targetDir will be rm -rf'ed when "make clean"
targetDir ?= ./generated
# The TEST_OUTPUT_DIR will be rm -rf'ed when "make clean"
TEST_OUTPUT_DIR ?= ./test-outputs
RM_DIRS 	:= $(TEST_OUTPUT_DIR) test-reports $(targetDir)
#CLEAN_DIRS	:= doc

test_src_dir := src/test/scala/ChiselTests
test_results := $(filter-out main DirChange Pads SIntOps,$(notdir $(basename $(wildcard $(test_src_dir)/*.scala))))
c_resources_dir := src/main/resources

test_outs    := $(addprefix $(targetDir)/, $(addsuffix .out, $(test_results)))

.PHONY:	smoke publish-local pubishLocal check clean jenkins-build coverage scaladoc test checkstyle compile

default:	publishLocal

smoke compile:
	$(SBT) $(SBT_FLAGS) compile

publish-local publishLocal:
	$(SBT) $(SBT_FLAGS) publishLocal

test:
	$(SBT) $(SBT_FLAGS) test

check:	test $(test_outs)

checkstyle:
	$(SBT) $(SBT_FLAGS) scalastyle test:scalastyle

coverage:
	$(SBT) $(SBT_FLAGS) coverage test
	$(SBT) $(SBT_FLAGS) coverageReport coverageAggregate

clean:
	$(SBT) $(SBT_FLAGS) clean
ifneq (,$(CLEAN_DIRS))
	for dir in $(CLEAN_DIRS); do $(MAKE) -C $$dir clean; done
endif
ifneq (,$(RM_DIRS))
	$(RM) -r $(RM_DIRS)
endif

scaladoc:
	$(SBT) $(SBT_FLAGS) unidoc

site:
	$(SBT) $(SBT_FLAGS) make-site

# Start off clean, then run tests for all supported configurations, and publish those versions of the code.
# Then run coverage and style tests (for developer's use).
# Don't publish the coverage test code since it contains hooks/references to the coverage test package
# and we don't want code with those dependencies published.
# We need to run the coverage tests last, since Jenkins will fail the build if it can't find their results.
jenkins-build: clean
	$(SBT) $(SBT_FLAGS) test
	$(SBT) $(SBT_FLAGS) clean publishLocal
	$(SBT) $(SBT_FLAGS) scalastyle coverage test
	$(SBT) $(SBT_FLAGS) coverageReport

$(targetDir)/%.fir: $(test_src_dir)/%.scala
	$(SBT) $(SBT_FLAGS) "test:runMain ChiselTests.MiniChisel $(notdir $(basename $<)) $(CHISEL_FLAGS)"

$(targetDir)/%.flo: $(targetDir)/%.fir
	$(CHISEL_BIN)/fir2flo.sh $(targetDir)/$*

$(targetDir)/%: $(targetDir)/%.flo $(targetDir)/emulator.h $(targetDir)/emulator_mod.h $(targetDir)/emulator_api.h
	(cd $(targetDir); $(CHISEL_BIN)/flo2app.sh $*)

$(targetDir)/%.h:	$(c_resources_dir)/%.h
	cp $< $@

$(targetDir)/%.out:	$(targetDir)/%
	$(SBT) $(SBT_FLAGS) "test:runMain ChiselTests.MiniChisel $(notdir $(basename $<)) $(CHISEL_FLAGS) --test --targetDir $(targetDir)"

# The "last-resort" rule.
# We assume the target is something like "+clean".
%::
	$(SBT) $(SBT_FLAGS) $@
