##===- CMakeLists.txt - MSFT dialect build definitions --------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect(MSFT msft)
add_circt_dialect_doc(MSFT msft)

set(LLVM_TARGET_DEFINITIONS MSFT.td)
mlir_tablegen(MSFTEnums.h.inc -gen-enum-decls)
mlir_tablegen(MSFTEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(MSFTAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=msft)
mlir_tablegen(MSFTAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=msft)

set(LLVM_TARGET_DEFINITIONS MSFTOpInterfaces.td)
mlir_tablegen(MSFTOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(MSFTOpInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(MLIRMSFTExtraIncGen)

set(LLVM_TARGET_DEFINITIONS MSFTPasses.td)
mlir_tablegen(MSFTPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTMSFTTransformsIncGen)
add_circt_doc(MSFTPasses MSFTPasses -gen-pass-doc)

add_dependencies(circt-headers MLIRMSFTExtraIncGen)
