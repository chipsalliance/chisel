add_custom_target(CIRCTUnitTests)
set_target_properties(CIRCTUnitTests PROPERTIES FOLDER "CIRCT Tests")

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()

function(add_circt_unittest test_dirname)
  add_unittest(CIRCTUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)
add_subdirectory(Support)
