##===- CMakeLists.txt - Object Model dialect build definitions ------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect(OM om)
add_circt_dialect_doc(OM om)

set(LLVM_TARGET_DEFINITIONS OM.td)
mlir_tablegen(OMAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(OMAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIROMAttrIncGen)
add_dependencies(circt-headers MLIROMAttrIncGen)
add_circt_interface(OMOpInterfaces)

set(LLVM_TARGET_DEFINITIONS OMPasses.td)
mlir_tablegen(OMPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTOMTransformsIncGen)
add_circt_doc(OMPasses OMPasses -gen-pass-doc)
