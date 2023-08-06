/*===- om.c - Simple test of OM C APIs ------------------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: circt-capi-om-test 2>&1 | FileCheck %s
 */

#include "circt-c/Dialect/OM.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include <mlir-c/Support.h>
#include <stdio.h>

void testEvaluator(MlirContext ctx) {
  const char *testIR =
      "module {"
      "  om.class @Test(%param: i8) {"
      "    om.class.field @field, %param : i8"
      "    %0 = om.object @Child() : () -> !om.class.type<@Child>"
      "    om.class.field @child, %0 : !om.class.type<@Child>"
      "  }"
      "  om.class @Child() {"
      "    %0 = om.constant 14 : i64"
      "    om.class.field @foo, %0 : i64"
      "  }"
      "}";

  // Set up the Evaluator.
  MlirModule testModule =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(testIR));

  OMEvaluator evaluator = omEvaluatorNew(testModule);

  MlirAttribute className =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("Test"));

  // Test instantiation failure.
  OMObject failedObject = omEvaluatorInstantiate(evaluator, className, 0, 0);

  // CHECK: error: actual parameter list length (0) does not match
  // CHECK: object is null: 1
  fprintf(stderr, "object is null: %d\n",
          omEvaluatorObjectIsNull(failedObject));

  // Test instantiation success.

  MlirAttribute actualParam =
      mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 8), 42);

  OMObject object =
      omEvaluatorInstantiate(evaluator, className, 1, &actualParam);

  // Test Object type.

  MlirType objectType = omEvaluatorObjectGetType(object);

  // CHECK: !om.class.type<@Test>
  mlirTypeDump(objectType);

  bool isClassType = omTypeIsAClassType(objectType);

  // CHECK: object type is class type: 1
  fprintf(stderr, "object type is class type: %d\n", isClassType);

  // Test get field failure.

  MlirAttribute missingFieldName =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("foo"));

  OMObjectValue missingField =
      omEvaluatorObjectGetField(object, missingFieldName);

  // CHECK: error: field "foo" does not exist
  // CHECK: field is null: 1
  fprintf(stderr, "field is null: %d\n",
          omEvaluatorObjectValueIsNull(missingField));

  // Test get field success.

  MlirAttribute fieldName =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("field"));

  OMObjectValue field = omEvaluatorObjectGetField(object, fieldName);

  // CHECK: field is object: 0
  fprintf(stderr, "field is object: %d\n",
          omEvaluatorObjectValueIsAObject(field));
  // CHECK: field is primitive: 1
  fprintf(stderr, "field is primitive: %d\n",
          omEvaluatorObjectValueIsAPrimitive(field));

  MlirAttribute fieldValue = omEvaluatorObjectValueGetPrimitive(field);

  // CHECK: 42 : i8
  mlirAttributeDump(fieldValue);

  // Test get field success for child object.

  MlirAttribute childFieldName =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("child"));

  OMObjectValue childField = omEvaluatorObjectGetField(object, childFieldName);

  MlirAttribute fieldNamesO = omEvaluatorObjectGetFieldNames(object);
  // CHECK: ["child", "field"]
  mlirAttributeDump(fieldNamesO);

  OMObject child = omEvaluatorObjectValueGetObject(childField);

  // CHECK: 0
  fprintf(stderr, "child object is null: %d\n", omEvaluatorObjectIsNull(child));

  OMObjectValue foo = omEvaluatorObjectGetField(
      child, mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("foo")));

  MlirAttribute fieldNamesC = omEvaluatorObjectGetFieldNames(child);

  // CHECK: ["foo"]
  mlirAttributeDump(fieldNamesC);

  // CHECK: child object field  is primitive: 1
  fprintf(stderr, "child object field is primitive: %d\n",
          omEvaluatorObjectValueIsAPrimitive(foo));

  MlirAttribute fooValue = omEvaluatorObjectValueGetPrimitive(foo);

  // CHECK: 14 : i64
  mlirAttributeDump(fooValue);
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__om__(), ctx);
  testEvaluator(ctx);
  return 0;
}
