SBT		?= sbt
SBT_FLAGS	?= -Dsbt.log.noformat=true

# If a chiselVersion is defined, use that.
# Otherwise, use the snapshot.
ifneq (,$(chiselVersion))
SBT_FLAGS += -DchiselVersion="$(chiselVersion)"
else
SBT_FLAGS += -DchiselVersion="3.0-SNAPSHOT"
endif

SRC_DIR	?= .
CHISEL_BIN ?= $(SRC_DIR)/bin
export CHISEL_BIN

SYSTEMC ?= $(SRC_DIR)/../../systemc/systemc-2.3.1
CHISEL_JAR ?= $(SRC_DIR)/target/scala-2.11/chisel_2.11-3.0-SNAPSHOT.jar
DRIVER	   ?= $(SRC_DIR)/src/test/resources/AddFilterSysCdriver.cpp
# The targetDir will be rm -rf'ed when "make clean"
targetDir ?= ./generated
# The TEST_OUTPUT_DIR will be rm -rf'ed when "make clean"
TEST_OUTPUT_DIR ?= ./test-outputs
RM_DIRS 	:= $(TEST_OUTPUT_DIR) test-reports $(targetDir)
#CLEAN_DIRS	:= doc

test_src_dir := src/test/scala/ChiselTests
test_results := $(filter-out main,$(notdir $(basename $(wildcard $(test_src_dir)/*.scala))))
c_resources_dir := src/main/resources

test_outs    := $(addprefix $(targetDir)/, $(addsuffix .out, $(test_results)))

.PHONY:	smoke publish-local check clean jenkins-build sysctest coverage scaladoc test

default:	publish-local

smoke:
	$(SBT) $(SBT_FLAGS) compile

publish-local:
	$(SBT) $(SBT_FLAGS) +publish-local

test:
	$(SBT) $(SBT_FLAGS) test

check:	test $(test_outs)

coverage:
	$(SBT) $(SBT_FLAGS) coverage test
	$(SBT) $(SBT_FLAGS) coverageReport

clean:
	$(SBT) $(SBT_FLAGS) +clean
ifneq (,$(CLEAN_DIRS))
	for dir in $(CLEAN_DIRS); do $(MAKE) -C $$dir clean; done
endif
ifneq (,$(RM_DIRS))
	$(RM) -r $(RM_DIRS)
endif

scaladoc:
	$(SBT) $(SBT_FLAGS) doc test:doc

# Start off clean, then run tests for all supported configurations, and publish those versions of the code.
# Then run coverage and style tests (for developer's use).
# Don't publish the coverage test code since it contains hooks/references to the coverage test package
# and we don't want code with those dependencies published.
# We need to run the coverage tests last, since Jenkins will fail the build if it can't find their results.
jenkins-build: clean
	$(SBT) $(SBT_FLAGS) +test
	$(SBT) $(SBT_FLAGS) +clean +publish-local
	$(SBT) $(SBT_FLAGS) scalastyle coverage test
	$(SBT) $(SBT_FLAGS) coverageReport

sysctest:
	mkdir -p $(TEST_OUTPUT_DIR)
	$(MAKE) -C $(TEST_OUTPUT_DIR) -f ../Makefile SRC_DIR=.. syscbuildandruntest

syscbuildandruntest:	AddFilter
	./AddFilter

AddFilter:	AddFilter.h AddFilter.cpp $(SYSC_DRIVER)
	$(CXX)  AddFilter.cpp $(DRIVER) \
	   -I. -I$(SYSTEMC)/include -L$(SYSTEMC)/lib-macosx64 -lsystemc -o $@

AddFilter.cpp AddFilter.h:	   AddFilter.class
	scala -cp $(CHISEL_JAR):. AddFilter --targetDir . --genHarness --backend sysc --design AddFilter

AddFilter.class:  $(CHISEL_JAR) ../src/test/scala/AddFilter.scala
	scalac -cp $(CHISEL_JAR) ../src/test/scala/AddFilter.scala

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
