add_circt_library(CIRCTReduceLib
  GenericReductions.cpp
  Reduction.cpp
  ReductionUtils.cpp
  Tester.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRReduceLib

  # Dialect dependencies.
  CIRCTFIRRTL
  CIRCTHW
)
