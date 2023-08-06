set(LLVM_OPTIONAL_SOURCES
    State.cpp
    Engine.cpp
    signals-runtime-wrappers.cpp
    Trace.cpp
)

add_circt_library(CIRCTLLHDSimState
    State.cpp
)

add_circt_library(CIRCTLLHDSimTrace
    Trace.cpp

    LINK_LIBS PUBLIC
    CIRCTLLHDSimState
)

add_circt_library(circt-llhd-signals-runtime-wrappers SHARED
    signals-runtime-wrappers.cpp

    LINK_LIBS PUBLIC
    CIRCTLLHDSimState
)
set_target_properties(circt-llhd-signals-runtime-wrappers
  PROPERTIES CXX_VISIBILITY_PRESET "default")

add_circt_library(CIRCTLLHDSimEngine
    Engine.cpp

    LINK_LIBS PUBLIC
    CIRCTLLHD
    CIRCTLLHDToLLVM
    CIRCTLLHDSimState
    CIRCTLLHDSimTrace
    circt-llhd-signals-runtime-wrappers
    MLIRExecutionEngine
    )
