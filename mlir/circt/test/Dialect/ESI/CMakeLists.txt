##===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(esi-tester
 esi-tester.cpp
)
llvm_update_compile_flags(esi-tester)
target_link_libraries(esi-tester
  PRIVATE
  CIRCTESI
  CIRCTHW
  CIRCTSV

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRFuncDialect
  MLIRTransforms
  MLIRLLVMDialect
  )
