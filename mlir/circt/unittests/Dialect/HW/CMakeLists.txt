add_circt_unittest(CIRCTHWTests
  HWModuleTest.cpp
  InstanceGraphTest.cpp
)

target_link_libraries(CIRCTHWTests
  PRIVATE
  CIRCTHW
)
