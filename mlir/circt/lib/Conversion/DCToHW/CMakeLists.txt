add_circt_library(CIRCTDCToHW
    DCToHW.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/DCToHW

    LINK_LIBS PUBLIC
    CIRCTHW
    CIRCTDC
    CIRCTESI
    CIRCTDC
    CIRCTDCTransforms
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRSupport
    MLIRTransforms
)
