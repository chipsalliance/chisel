add_circt_unittest(CIRCTFIRRTLTests
  TypesTest.cpp
)

target_link_libraries(CIRCTFIRRTLTests
  PRIVATE
  CIRCTFIRRTL
)
