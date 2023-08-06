//===- TypesTest.cpp - FIRRTL type unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {

TEST(TypesTest, AnalogContainsAnalog) {
  MLIRContext context;
  context.loadDialect<FIRRTLDialect>();
  ASSERT_TRUE(AnalogType::get(&context).containsAnalog());
}

TEST(TypesTest, TypeAliasCast) {
  MLIRContext context;
  context.loadDialect<FIRRTLDialect>();
  // Check ContainAliasableTypes.
  static_assert(!ContainAliasableTypes<FIRRTLType>::value);
  // Return false for FIRRTLBaseType.
  static_assert(!ContainAliasableTypes<FIRRTLBaseType>::value);
  static_assert(!ContainAliasableTypes<StringType>::value);
  static_assert(ContainAliasableTypes<FVectorType>::value);
  static_assert(ContainAliasableTypes<UIntType, StringType>::value);
  static_assert(ContainAliasableTypes<hw::FieldIDTypeInterface>::value);
  AnalogType analog = AnalogType::get(&context);
  BaseTypeAliasType alias1 =
      BaseTypeAliasType::get(StringAttr::get(&context, "foo"), analog);
  BaseTypeAliasType alias2 =
      BaseTypeAliasType::get(StringAttr::get(&context, "bar"), alias1);
  ASSERT_TRUE(!type_isa<FVectorType>(analog));
  ASSERT_TRUE(type_isa<AnalogType>(analog));
  ASSERT_TRUE(type_isa<AnalogType>(alias1));
  ASSERT_TRUE(type_isa<AnalogType>(alias2));
  ASSERT_TRUE(!type_isa<FVectorType>(alias2));
  ASSERT_TRUE((type_isa<AnalogType, StringType>(alias2)));
}

} // namespace
