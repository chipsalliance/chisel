##===- CMakeLists.txt - Calyx emitter code ---------------------*- cmake -*-===//
##
## Implementation files for the Calyx emitter.
##
##===----------------------------------------------------------------------===//

set(srcs
  CalyxEmitter.cpp
  )

add_circt_dialect_library(CIRCTExportCalyx
  ${srcs}
  LINK_LIBS PUBLIC
  CIRCTCalyx
  MLIRTranslateLib
  )
