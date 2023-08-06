//===- JSONTest.cpp - JSON encoder/decoder unit tests ===------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/JSON.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace circt;
using namespace mlir;

namespace json = llvm::json;

namespace {

TEST(JSONTest, RoundTripTypes) {
  MLIRContext context;

  auto i64ty = IntegerType::get(&context, 64);

  NamedAttrList mapping;

  auto floatAttr = FloatAttr::get(FloatType::getF64(&context), 123.4);
  mapping.append("float", floatAttr);

  auto intAttr = IntegerAttr::get(i64ty, 567);
  mapping.append("int", intAttr);

  auto boolAttr = BoolAttr::get(&context, true);
  mapping.append("bool", boolAttr);

  auto stringAttr = StringAttr::get(&context, "bohooo");
  mapping.append("string", stringAttr);

  ArrayAttr arrayAttr;
  {
    SmallVector<Attribute> array;
    array.push_back(IntegerAttr::get(i64ty, 1));
    array.push_back(IntegerAttr::get(i64ty, 2));
    arrayAttr = ArrayAttr::get(&context, array);

    mapping.append("array", arrayAttr);
  }

  {
    NamedAttrList dict;
    dict.append("x", IntegerAttr::get(i64ty, 1));
    dict.append("y", IntegerAttr::get(i64ty, 2));
    mapping.append("dict", dict.getDictionary(&context));
  }

  DictionaryAttr sampleDict = mapping.getDictionary(&context);

  std::string jsonBuffer;
  llvm::raw_string_ostream jsonOs(jsonBuffer);
  llvm::json::OStream json(jsonOs, 2);
  ASSERT_TRUE(succeeded(convertAttributeToJSON(json, sampleDict)));

  auto jsonValue = json::parse(jsonBuffer);
  ASSERT_TRUE(bool(jsonValue));

  json::Path::Root root;
  auto dict = convertJSONToAttribute(&context, jsonValue.get(), root)
                  .cast<DictionaryAttr>();

  ASSERT_EQ(floatAttr, dict.getAs<FloatAttr>("float"));
  ASSERT_EQ(intAttr, dict.getAs<IntegerAttr>("int"));
  ASSERT_EQ(boolAttr, dict.getAs<BoolAttr>("bool"));
  ASSERT_EQ(stringAttr, dict.getAs<StringAttr>("string"));
  ASSERT_EQ(arrayAttr, dict.getAs<ArrayAttr>("array"));

  auto dictField = dict.getAs<DictionaryAttr>("dict");
  EXPECT_EQ(1, dictField.getAs<IntegerAttr>("x").getValue());
  EXPECT_EQ(2, dictField.getAs<IntegerAttr>("y").getValue());
}

} // namespace
