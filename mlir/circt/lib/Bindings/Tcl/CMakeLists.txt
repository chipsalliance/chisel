add_library(circt-tcl
  SHARED
  circt_tcl.cpp
)
target_link_libraries(circt-tcl
  PUBLIC
  CIRCTCAPIComb
  CIRCTCAPIESI
  CIRCTCAPIMSFT
  CIRCTCAPIHW
  CIRCTCAPISeq
  CIRCTCAPISV
  CIRCTCAPIFSM
)
target_compile_definitions(circt-tcl PUBLIC -DUSE_TCL_STUBS)
target_link_libraries(circt-tcl PUBLIC ${TCL_STUB_LIBRARY})
target_include_directories(circt-tcl PUBLIC ${TCL_INCLUDE_PATH})
