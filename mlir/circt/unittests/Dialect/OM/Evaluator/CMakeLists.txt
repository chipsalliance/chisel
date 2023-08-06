add_circt_unittest(CIRCTOMEvaluatorTests
  EvaluatorTests.cpp
)

target_link_libraries(CIRCTOMEvaluatorTests
  PRIVATE
  CIRCTOMEvaluator
  CIRCTOM
  MLIRIR
)
