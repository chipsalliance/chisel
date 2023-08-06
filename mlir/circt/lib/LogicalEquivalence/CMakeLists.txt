if(CIRCT_LEC_ENABLED)
  add_circt_verification_library(CIRCTLogicalEquivalence
    Solver.cpp
    Circuit.cpp
    LogicExporter.cpp

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRTransforms
    MLIRTranslateLib
    CIRCTComb
    CIRCTHW
    CIRCTSupport
  )

  target_link_libraries(CIRCTLogicalEquivalence
    PUBLIC
    ${Z3_LIBRARIES}
  )

  target_include_directories(CIRCTLogicalEquivalence SYSTEM
    PUBLIC
    ${Z3_INCLUDE_DIR}
    ${Z3_CXX_INCLUDE_DIRS}
  )
endif()
