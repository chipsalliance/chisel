//===- MemoryOps.td - LLHD memory operations ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for LLHD memory allocation and access.
//
//===----------------------------------------------------------------------===//

def LLHD_VarOp : LLHD_Op<"var", [
    TypesMatchWith<
      "type of 'init' and underlying type of 'result' have to match.",
      "init", "result", "PtrType::get($_self)">
  ]> {
  let summary = "Stack allocation.";
  let description = [{
    The `llhd.var` operation allocates a memory region on the stack with the
    initial value given by `init`, and returns a pointer to the allocated
    region.

    **Examples:**

    ```
    %int = llhd.const 0 : i32
    %arr = llhd.array_uniform %int : !llhd.array<3xi32>

    %iPtr = llhd.var %int : i32
    %arrPtr = llhd.var %arr : !llhd.array<3xi32>
    ```
  }];

  let arguments = (ins LLHD_AnyUnderlyingType: $init);
  let results = (outs Res<LLHD_AnyPtrType, "the newly allocated reference",
                          [MemAlloc]>: $result);

  let assemblyFormat = "$init attr-dict `:` qualified(type($init))";
}

def LLHD_LoadOp : LLHD_Op<"load", [
    TypesMatchWith<
      "type of 'result' and underlying type of 'pointer' have to match.",
      "pointer", "result", "$_self.cast<PtrType>().getUnderlyingType()">
  ]> {
  let summary = "Load a value.";
  let description = [{
    The `llhd.load` operation loads a value from a memory region given by
    `pointer`.

    **Examples:**

    ```
    %int = llhd.const 0 : i32
    %arr = llhd.array_uniform %int : !llhd.array<3xi32>
    %iPtr = llhd.var %int : i32
    %arrPtr = llhd.var %arr : !llhd.array<3xi32>

    %iLd = llhd.load %iPtr : !llhd.ptr<i32>
    %arrLd = llhd.load %arrPtr : !llhd.ptr<!llhd.array<3xi32>>
    ```
  }];

  let arguments = (ins Arg<LLHD_AnyPtrType, "the reference to load from",
                           [MemRead]>: $pointer);
  let results = (outs LLHD_AnyUnderlyingType: $result);

  let assemblyFormat = "$pointer attr-dict `:` qualified(type($pointer))";
}

def LLHD_StoreOp : LLHD_Op<"store", [
    TypesMatchWith<
      "type of 'value' and underlying type of 'pointer' have to match.",
      "pointer", "value", "$_self.cast<PtrType>().getUnderlyingType()">
  ]> {
  let summary = "Store a value.";
  let description = [{
    The `llhd.store` operation stores the value `value` to the memory region
    given by `pointer`.

    **Examples:**

    ```
    %int = llhd.const 0 : i32
    %arr = llhd.array_uniform %int : !llhd.array<3xi32>
    %iPtr = llhd.var %int : i32
    %arrPtr = llhd.var %arr : !llhd.array<3xi32>

    llhd.store %iPtr, %int : !llhd.ptr<i32>
    llhd.store %arrPtr, %arr : !llhd.ptr<!llhd.array<3xi32>>
    ```
  }];

  let arguments = (ins Arg<LLHD_AnyPtrType, "the reference to store to",
                           [MemWrite]>: $pointer,
                       LLHD_AnyUnderlyingType: $value);

  let assemblyFormat = "$pointer `,` $value attr-dict `:` qualified(type($pointer))";
}
