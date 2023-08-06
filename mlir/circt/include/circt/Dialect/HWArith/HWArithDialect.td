//===- HWArithDialect.td - HWArith dialect definition ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the HWArithDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef HWARITHDIALECT_TD
#define HWARITHDIALECT_TD

def HWArithDialect : Dialect {
  let name = "hwarith";
  let cppNamespace = "::circt::hwarith";

  let summary = "Types and operations for the HWArith dialect";
  let description = [{
    This dialect defines the `HWArith` dialect, modeling bit-width aware
    arithmetic operations.
  }];
}

#endif // HWARITHDIALECT_TD
