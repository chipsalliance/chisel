add_circt_unittest(CIRCTSupportTests
  JSONTest.cpp
  PrettyPrinterTest.cpp
)

target_link_libraries(CIRCTSupportTests
  PRIVATE
  CIRCTSupport
)
