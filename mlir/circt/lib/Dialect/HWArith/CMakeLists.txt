# ===- CMakeLists.txt - HWArith dialect code ------------------*- cmake -*-===//
#
# Implementation files for the HWArith dialect.
#
# ===----------------------------------------------------------------------===//

set(srcs
    HWArithDialect.cpp
    HWArithOps.cpp
    HWArithTypes.cpp
)

add_circt_dialect_library(CIRCTHWArith
    ${srcs}

    DEPENDS
    MLIRHWArithIncGen
    MLIRHWArithEnumsIncGen
    MLIRHWArithCanonicalizationsIncGen
    MLIRIR

    LINK_COMPONENTS
    Core
    Support

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)
