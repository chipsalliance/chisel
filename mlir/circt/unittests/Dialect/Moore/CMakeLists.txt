add_circt_unittest(CIRCTMooreTests
  TypesTest.cpp
)

target_link_libraries(CIRCTMooreTests
  PRIVATE
  CIRCTMoore
)
