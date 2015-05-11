SBT		?= sbt
SBT_FLAGS	?= -Dsbt.log.noformat=true
RM_DIRS 	:= test-outputs test-reports
CLEAN_DIRS	:= doc

SRC_DIR	?= .
SYSTEMC ?= $(SRC_DIR)/../../systemc/systemc-2.3.1
CHISEL_JAR ?= $(SRC_DIR)/target/scala-2.11/chisel_2.11-3.0-SNAPSHOT.jar
DRIVER	   ?= $(SRC_DIR)/src/test/resources/AddFilterSysCdriver.cpp
TEST_OUTPUT_DIR ?= ./test-outputs

.PHONY:	smoke publish-local check clean jenkins-build sysctest coverage scaladoc test

default:	publish-local

smoke:
	$(SBT) $(SBT_FLAGS) compile

publish-local:
	$(SBT) $(SBT_FLAGS) publish-local

check test:
	$(SBT) $(SBT_FLAGS) test

coverage:
	$(SBT) $(SBT_FLAGS) coverage test
	$(SBT) $(SBT_FLAGS) coverageReport

clean:
	$(SBT) $(SBT_FLAGS) +clean
	for dir in $(CLEAN_DIRS); do $(MAKE) -C $$dir clean; done
	$(RM) -r $(RM_DIRS)

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

