//===- CalyxPrimitives.td - Calyx Primitives -------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the primitives in the Calyx dialect.
//
//===----------------------------------------------------------------------===//

/// Base class for Calyx primitives.
class CalyxPrimitive<string mnemonic, list<Trait> traits = []> :
  CalyxCell<mnemonic, traits> {
    let assemblyFormat = "$sym_name attr-dict `:` qualified(type(results))";
    let arguments = (ins SymbolNameAttr:$sym_name);
    let skipDefaultBuilders = 1;
}

/// Trait used to mark certain operations in the Calyx dialect as combinational.
def Combinational : NativeOpTrait<"Combinational"> {
  let cppNamespace = "::circt::calyx";
}

def RegisterOp : CalyxPrimitive<"register", [
    SameTypeConstraint<"in", "out">
  ]> {
  let summary = "Defines a register";
  let description = [{
    The "calyx.register" op defines a register.
    ```mlir
      // A 32-bit register.
      %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
    ```
  }];
  let results = (outs AnyType:$in, I1:$write_en, I1:$clk, I1:$reset, AnyType:$out, I1:$done);
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "size_t":$width), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(sym_name));
      Type i1Type = $_builder.getI1Type();
      Type widthType = $_builder.getIntegerType(width);
      $_state.addTypes({widthType, i1Type, i1Type, i1Type, widthType, i1Type});
    }]>,
    OpBuilder<(ins "StringRef":$sym_name, "Type":$type), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(sym_name));
      Type i1Type = $_builder.getI1Type();
      $_state.addTypes({type, i1Type, i1Type, i1Type, type, i1Type});
    }]>
  ];
}

def MemoryOp : CalyxPrimitive<"memory", []> {
  let summary = "Defines a memory";
  let description = [{
    The "calyx.memory" op defines a memory. Memories can have any number of
    dimensions, as specified by the length of the `$sizes` and `$addrSizes`
    arrays. The `$addrSizes` specify the bitwidth of each dimension's address,
    and should be wide enough to address the range of the corresponding
    dimension in `$sizes`. The `$width` attribute dictates the width of a single
    element.

    See https://docs.calyxir.org/libraries/core.html#memories for
    more information.

    ```mlir
      // A 1-dimensional, 32-bit memory with size dimension 1. Equivalent representation in the native compiler:
      // `m1 = std_mem_d1(32, 1, 1)`
      %m1.addr0, %m1.write_data, %m1.write_en, %m1.clk, %m1.read_data, %m1.done = calyx.memory @m1 <[1] x 32> [1] : i1, i32, i1, i1, i32, i1

      // A 2-dimensional, 8-bit memory with size dimensions 64 x 64. Equivalent representation in the native compiler:
      // `m2 = std_mem_d2(8, 64, 64, 6, 6)`
      %m2.addr0, %m2.addr1, %m2.write_data, %m2.write_en, %m2.clk, %m2.read_data, %m2.done = calyx.memory @m2 <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$width,
    ArrayAttr:$sizes,
    ArrayAttr:$addrSizes
  );

  let results = (outs
    Variadic<AnySignlessInteger>:$results
  );

  let assemblyFormat = [{
    $sym_name ` ` `<` $sizes `x` $width `>` $addrSizes attr-dict `:` qualified(type($results))
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "int64_t":$width,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$addrSizes
    )>
  ];

  let extraClassDeclaration = [{
    size_t getNumDimensions() { return getAddrSizes().size(); }

    ValueRange addrPorts() { return getResults().take_front(getNumDimensions()); }
    Value addrPort(size_t i) {
      assert(i < getNumDimensions() && "index greater than number of memory address ports.");
      return getResults()[i];
    }
    Value writeData() { return getResult(getNumDimensions()); }
    Value writeEn()   { return getResult(getNumDimensions() + 1); }
    Value clk()       { return getResult(getNumDimensions() + 2); }
    Value readData()  { return getResult(getNumDimensions() + 3); }
    Value done()      { return getResult(getNumDimensions() + 4); }
  }];

  let extraClassDefinition = [{
    bool $cppClass::isCombinational() { return false; }
  }];
}

def SeqMemoryOp : CalyxPrimitive<"seq_mem", []> {
  let summary = "Defines a memory with sequential read";
  let description = [{
    The "calyx.seq_mem" op defines a memory with sequential reads. Memories can 
    have any number of dimensions, as specified by the length of the `$sizes` and
    `$addrSizes` arrays. The `$addrSizes` specify the bitwidth of each dimension's
    address, and should be wide enough to address the range of the corresponding
    dimension in `$sizes`. The `$width` attribute dictates the width of a single
    element.

    See https://docs.calyxir.org/libraries/core.html#memories for
    more information.

    ```mlir
      // A 1-dimensional, 32-bit memory with size dimension 1. Equivalent representation in the native compiler:
      // `m1 = seq_mem_d1(32, 1, 1)`
      %m1.addr0, %m1.write_data, %m1.write_en, %m1.write_done, %m1.clk, %m1.read_data, %m1.read_en, %m1.read_done = calyx.memory @m1 <[1] x 32> [1] : i1, i32, i1, i1, i32, i1

      // A 2-dimensional, 8-bit memory with size dimensions 64 x 64. Equivalent representation in the native compiler:
      // `m2 = seq_mem_d2(8, 64, 64, 6, 6)`
      %m2.addr0, %m2.addr1, %m2.write_data, %m2.write_en, %m2.write_done, %m2.clk, %m2.read_data, %m2.read_en, %m2.read_done = calyx.memory @m2 <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$width,
    ArrayAttr:$sizes,
    ArrayAttr:$addrSizes
  );

  let results = (outs
    Variadic<AnySignlessInteger>:$results
  );

  let assemblyFormat = [{
    $sym_name ` ` `<` $sizes `x` $width `>` $addrSizes attr-dict `:` qualified(type($results))
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins
      "StringRef":$sym_name,
      "int64_t":$width,
      "ArrayRef<int64_t>":$sizes,
      "ArrayRef<int64_t>":$addrSizes
    )>
  ];

  let extraClassDeclaration = [{
    size_t getNumDimensions() { return getAddrSizes().size(); }

    ValueRange addrPorts() { return getResults().take_front(getNumDimensions()); }
    Value addrPort(size_t i) {
      assert(i < getNumDimensions() && "index greater than number of memory address ports.");
      return getResults()[i];
    }
    Value writeData() { return getResult(getNumDimensions()); }
    Value writeEn()   { return getResult(getNumDimensions() + 1); }
    Value writeDone()      { return getResult(getNumDimensions() + 2); }
    Value clk()       { return getResult(getNumDimensions() + 3); }
    Value readData()  { return getResult(getNumDimensions() + 4); }
    Value readEn()  { return getResult(getNumDimensions() + 5); }
    Value readDone()      { return getResult(getNumDimensions() + 6); }
  }];

  let extraClassDefinition = [{
    bool $cppClass::isCombinational() { return false; }
  }];
}

class CalyxLibraryOp<string mnemonic, list<Trait> traits = []> :
  CalyxPrimitive<"std_" # mnemonic, traits> {

  let summary = "Defines an operation which maps to a Calyx library primitive";
  let description = [{
    This operation represents an instance of a Calyx library primitive.
    A library primitive maps to some hardware-implemented component within the
    native Calyx compiler.

    ```mlir
      // A 32-bit adder. This falls under the binary library operations.
      %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

      // An 8-bit comparison operator (with a 1-bit output). This falls under
      // the boolean binary library operations.
      %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1

      // An 8-bit to 16-bit pad operator. This falls under the unary
      // library operations.
      %pad.in, %pad.out = calyx.std_pad @pad : i8, i16
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "::mlir::TypeRange":$resultTypes), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(sym_name));
      $_state.addTypes(resultTypes);
    }]>
  ];
}

class BoolBinaryLibraryOp<string mnemonic> : CalyxLibraryOp<mnemonic, [
    Combinational,
    SameTypeConstraint<"left", "right">
  ]> {
  let results = (outs AnyType:$left, AnyType:$right, I1:$out);
}

def LtLibOp  : BoolBinaryLibraryOp<"lt"> {}
def GtLibOp  : BoolBinaryLibraryOp<"gt"> {}
def EqLibOp  : BoolBinaryLibraryOp<"eq"> {}
def NeqLibOp : BoolBinaryLibraryOp<"neq"> {}
def GeLibOp  : BoolBinaryLibraryOp<"ge"> {}
def LeLibOp  : BoolBinaryLibraryOp<"le"> {}
def SltLibOp  : BoolBinaryLibraryOp<"slt"> {}
def SgtLibOp  : BoolBinaryLibraryOp<"sgt"> {}
def SeqLibOp  : BoolBinaryLibraryOp<"seq"> {}
def SneqLibOp : BoolBinaryLibraryOp<"sneq"> {}
def SgeLibOp  : BoolBinaryLibraryOp<"sge"> {}
def SleLibOp  : BoolBinaryLibraryOp<"sle"> {}

class ArithBinaryLibraryOp<string mnemonic, list<Trait> traits = []> :
  CalyxLibraryOp<mnemonic, !listconcat(traits, [
    SameTypeConstraint<"left", "right">
  ])> {}

class CombinationalArithBinaryLibraryOp<string mnemonic> :
  ArithBinaryLibraryOp<mnemonic, [
    Combinational,
    SameTypeConstraint<"left", "out">
  ]> {
  let results = (outs AnyType:$left, AnyType:$right, AnyType:$out);
}

def AddLibOp  : CombinationalArithBinaryLibraryOp<"add"> {}
def SubLibOp  : CombinationalArithBinaryLibraryOp<"sub"> {}
def ShruLibOp : CombinationalArithBinaryLibraryOp<"shru"> {}
def RshLibOp  : CombinationalArithBinaryLibraryOp<"rsh"> {}
def SrshLibOp : CombinationalArithBinaryLibraryOp<"srsh"> {}
def LshLibOp  : CombinationalArithBinaryLibraryOp<"lsh"> {}
def AndLibOp  : CombinationalArithBinaryLibraryOp<"and"> {}
def OrLibOp   : CombinationalArithBinaryLibraryOp<"or"> {}
def XorLibOp  : CombinationalArithBinaryLibraryOp<"xor"> {}

class ArithBinaryPipeLibraryOp<string mnemonic> : ArithBinaryLibraryOp<mnemonic # "_pipe", [
    SameTypeConstraint<"left", "out">
  ]> {
  let results = (outs I1:$clk, I1:$reset, I1:$go, AnyType:$left, AnyType:$right, AnyType:$out, I1:$done);
}

def MultPipeLibOp : ArithBinaryPipeLibraryOp<"mult"> {}
def DivSPipeLibOp : ArithBinaryPipeLibraryOp<"divs"> {}
def DivUPipeLibOp : ArithBinaryPipeLibraryOp<"divu"> {}
def RemUPipeLibOp : ArithBinaryPipeLibraryOp<"remu"> {}
def RemSPipeLibOp : ArithBinaryPipeLibraryOp<"rems"> {}

class UnaryLibraryOp<string mnemonic, list<Trait> traits = []> :
    CalyxLibraryOp<mnemonic, !listconcat(traits, [Combinational])> {
  let results = (outs AnyInteger:$in, AnyInteger:$out);
}

def PadLibOp : UnaryLibraryOp<"pad"> {
  let hasVerifier = 1;
}

def SliceLibOp : UnaryLibraryOp<"slice"> {
  let hasVerifier = 1;
}

def NotLibOp  : UnaryLibraryOp<"not"> {}

def WireLibOp : UnaryLibraryOp<"wire", [SameTypeConstraint<"in", "out">]> {
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "::mlir::Type":$wireType), [{
      llvm::SmallVector<mlir::Type> resultTypes = {wireType, wireType};
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), $_builder.getStringAttr(sym_name));
      $_state.addTypes(resultTypes);
    }]>
  ];
}

def ExtSILibOp  : UnaryLibraryOp<"extsi"> {}
