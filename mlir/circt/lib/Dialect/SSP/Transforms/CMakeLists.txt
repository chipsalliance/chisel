add_circt_dialect_library(CIRCTSSPTransforms
  Print.cpp
  Roundtrip.cpp
  Schedule.cpp

  DEPENDS
  CIRCTSSPTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTScheduling
  CIRCTSSP
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

if(ortools_FOUND)
  target_compile_definitions(obj.CIRCTSSPTransforms PRIVATE SCHEDULING_OR_TOOLS)
endif()
