//===- EvaluatorTest.cpp - Object Model evaluator tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"
#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;
using namespace circt::om;

namespace {

/// Failure scenarios.

TEST(EvaluatorTests, InstantiateInvalidClassName) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "unknown class name \"MyClass\"");
  });

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_FALSE(succeeded(result));
}

TEST(EvaluatorTests, InstantiateInvalidParamSize) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  auto cls = builder.create<ClassOp>("MyClass", params);
  cls.getBody().emplaceBlock().addArgument(builder.getIntegerType(32),
                                           cls.getLoc());

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(
        diag.str(),
        "actual parameter list length (0) does not match formal parameter "
        "list length (1)");
  });

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_FALSE(succeeded(result));
}

TEST(EvaluatorTests, InstantiateNullParam) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  auto cls = builder.create<ClassOp>("MyClass", params);
  cls.getBody().emplaceBlock().addArgument(builder.getIntegerType(32),
                                           cls.getLoc());

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "actual parameter for \"param\" is null");
  });

  auto result =
      evaluator.instantiate(builder.getStringAttr("MyClass"), {IntegerAttr()});

  ASSERT_FALSE(succeeded(result));
}

TEST(EvaluatorTests, InstantiateInvalidParamType) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  auto cls = builder.create<ClassOp>("MyClass", params);
  cls.getBody().emplaceBlock().addArgument(builder.getIntegerType(32),
                                           cls.getLoc());

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "actual parameter for \"param\" has invalid type");
  });

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"),
                                      {builder.getF32FloatAttr(42)});

  ASSERT_FALSE(succeeded(result));
}

TEST(EvaluatorTests, GetFieldInvalidName) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass");
  cls.getBody().emplaceBlock();

  Evaluator evaluator(mod);

  context.getDiagEngine().registerHandler([&](Diagnostic &diag) {
    ASSERT_EQ(diag.str(), "field \"foo\" does not exist");
  });

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = result.value()->getField(builder.getStringAttr("foo"));

  ASSERT_FALSE(succeeded(fieldValue));
}

/// Success scenarios.

TEST(EvaluatorTests, InstantiateObjectWithParamField) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  StringRef fields[] = {"field"};
  Type types[] = {builder.getIntegerType(32)};
  ClassOp::buildSimpleClassOp(builder, loc, "MyClass", params, fields, types);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"),
                                      {builder.getI32IntegerAttr(42)});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue =
      std::get<Attribute>(
          result.value()->getField(builder.getStringAttr("field")).value())
          .dyn_cast<IntegerAttr>();
  ASSERT_TRUE(fieldValue);
  ASSERT_EQ(fieldValue.getValue(), 42);
}

TEST(EvaluatorTests, InstantiateObjectWithConstantField) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass");
  auto &body = cls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  auto constant = builder.create<ConstantOp>(builder.getI32IntegerAttr(42));
  builder.create<ClassFieldOp>("field", constant);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue =
      std::get<Attribute>(
          result.value()->getField(builder.getStringAttr("field")).value())
          .dyn_cast<IntegerAttr>();
  ASSERT_TRUE(fieldValue);
  ASSERT_EQ(fieldValue.getValue(), 42);
}

TEST(EvaluatorTests, InstantiateObjectWithChildObject) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  StringRef fields[] = {"field"};
  Type types[] = {builder.getIntegerType(32)};
  auto innerCls = ClassOp::buildSimpleClassOp(builder, loc, "MyInnerClass",
                                              params, fields, types);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass", params);
  auto &body = cls.getBody().emplaceBlock();
  body.addArgument(builder.getIntegerType(32), cls.getLoc());
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  builder.create<ClassFieldOp>("field", object);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"),
                                      {builder.getI32IntegerAttr(42)});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue = std::get<std::shared_ptr<Object>>(
      result.value()->getField(builder.getStringAttr("field")).value());

  ASSERT_TRUE(fieldValue);

  auto innerFieldValue =
      std::get<Attribute>(
          fieldValue->getField(builder.getStringAttr("field")).value())
          .cast<IntegerAttr>();

  ASSERT_EQ(innerFieldValue.getValue(), 42);
}

TEST(EvaluatorTests, InstantiateObjectWithFieldAccess) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  StringRef params[] = {"param"};
  StringRef fields[] = {"field"};
  Type types[] = {builder.getIntegerType(32)};
  auto innerCls = ClassOp::buildSimpleClassOp(builder, loc, "MyInnerClass",
                                              params, fields, types);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass", params);
  auto &body = cls.getBody().emplaceBlock();
  body.addArgument(builder.getIntegerType(32), cls.getLoc());
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  auto field =
      builder.create<ObjectFieldOp>(builder.getI32Type(), object,
                                    builder.getArrayAttr(FlatSymbolRefAttr::get(
                                        builder.getStringAttr("field"))));
  builder.create<ClassFieldOp>("field", field);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"),
                                      {builder.getI32IntegerAttr(42)});

  ASSERT_TRUE(succeeded(result));

  auto fieldValue =
      std::get<Attribute>(
          result.value()->getField(builder.getStringAttr("field")).value())
          .cast<IntegerAttr>();

  ASSERT_TRUE(fieldValue);
  ASSERT_EQ(fieldValue.getValue(), 42);
}

TEST(EvaluatorTests, InstantiateObjectWithChildObjectMemoized) {
  DialectRegistry registry;
  registry.insert<OMDialect>();

  MLIRContext context(registry);
  context.getOrLoadDialect<OMDialect>();

  Location loc(UnknownLoc::get(&context));

  ImplicitLocOpBuilder builder(loc, &context);

  auto mod = builder.create<ModuleOp>(loc);

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto innerCls = builder.create<ClassOp>("MyInnerClass");
  innerCls.getBody().emplaceBlock();

  builder.setInsertionPointToStart(&mod.getBodyRegion().front());
  auto cls = builder.create<ClassOp>("MyClass");
  auto &body = cls.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  auto object = builder.create<ObjectOp>(innerCls, body.getArguments());
  builder.create<ClassFieldOp>("field1", object);
  builder.create<ClassFieldOp>("field2", object);

  Evaluator evaluator(mod);

  auto result = evaluator.instantiate(builder.getStringAttr("MyClass"), {});

  ASSERT_TRUE(succeeded(result));

  auto field1Value = std::get<std::shared_ptr<Object>>(
      result.value()->getField(builder.getStringAttr("field1")).value());

  auto field2Value = std::get<std::shared_ptr<Object>>(
      result.value()->getField(builder.getStringAttr("field2")).value());

  auto fieldNames = result.value()->getFieldNames();

  ASSERT_TRUE(fieldNames.size() == 2);
  StringRef fieldNamesTruth[] = {"field1", "field2"};
  for (auto fieldName : llvm::enumerate(fieldNames)) {
    auto str = llvm::dyn_cast_or_null<StringAttr>(fieldName.value());
    ASSERT_TRUE(str);
    ASSERT_EQ(str.getValue(), fieldNamesTruth[fieldName.index()]);
  }

  ASSERT_TRUE(field1Value);
  ASSERT_TRUE(field2Value);

  ASSERT_EQ(field1Value, field2Value);
}

} // namespace
