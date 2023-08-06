//===- FIRRTLStatements.td - FIRRTL statement ops ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for FIRRTL statements.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLSTATEMENTS_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLSTATEMENTS_TD

include "FIRRTLDialect.td"
include "FIRRTLEnums.td"
include "FIRRTLOpInterfaces.td"
include "FIRRTLTypes.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AttachOp : FIRRTLOp<"attach"> {
  let summary = "Analog Attach Statement";

  let arguments = (ins Variadic<AnalogType>:$attached);
  let results = (outs);

  let assemblyFormat = "$attached attr-dict `:` qualified(type($attached))";

  let hasVerifier = 1;
  let hasCanonicalizeMethod = true;
}

def ConnectOp : FIRRTLOp<"connect", [FConnectLike]> {
  let summary = "Connect two signals";
  let description = [{
    Connect Operation:
    ```
      firrtl.connect %dest, %src : t1, t2
    ```
    }];

  let arguments = (ins ConnectableType:$dest, ConnectableType:$src);
  let results = (outs);

  let assemblyFormat =
    "$dest `,` $src  attr-dict `:` qualified(type($dest)) `,` qualified(type($src))";

  let hasVerifier = 1;
  let hasCanonicalizer = true;
}

def SameAnonTypeOperands: PredOpTrait<
        "operands must be structurally equivalent",
        CPred<"areAnonymousTypesEquivalent(getOperand(0).getType(), getOperand(1).getType())">>;
def StrictConnectOp : FIRRTLOp<"strictconnect", [FConnectLike,
                      SameAnonTypeOperands]> {
  let summary = "Connect two signals";
  let description =  [{
    Connect two values with strict constraints:
    ```
      firrtl.strictconnect %dest, %src : t1
      firrtl.strictconnect %dest, %src : t1, !firrtl.alias<foo, t1>
    ```
    }];

  let arguments = (ins StrictConnectableType:$dest,
                       StrictConnectableType:$src);
  let results = (outs);
  let hasCanonicalizeMethod = true;

  let hasVerifier = 1;
  let assemblyFormat = [{$dest `,` $src  attr-dict `:`
      custom<OptionalBinaryOpTypes>(type($dest), type($src))}];
}

def RefDefineOp : FIRRTLOp<"ref.define", [SameTypeOperands, FConnectLike]> {
  let summary = "FIRRTL Define References";
  let description = [{
    Define a target reference to the source reference:
    ```
      firrtl.ref.define %dest, %src : ref<t1>
    ```
    Used to statically route reference from source to destination
    through the design, one module at a time.

    Similar to "connect" but cannot have multiple define's to same
    destination and the define is never conditional even if under
    a "when".

    Source and destination must resolve statically.
  }];
  let arguments = (ins RefType:$dest, RefType:$src);

  let hasVerifier = 1;

  let assemblyFormat =
    "$dest `,` $src  attr-dict `:` qualified(type($dest))";

  let extraClassDeclaration = [{
    static ConnectBehaviorKind getConnectBehaviorKind() {
      return ConnectBehaviorKind::StaticSingleConnect;
    }
  }];
}


def PrintFOp : FIRRTLOp<"printf"> {
  let summary = "Formatted Print Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$cond, StrAttr:$formatString,
                       Variadic<FIRRTLBaseType>:$substitutions, StrAttr:$name);
  let results = (outs);

  let assemblyFormat = [{
    $clock `,` $cond `,` $formatString `` custom<PrintfAttrs>(attr-dict) ` `
    (`(` $substitutions^ `)`)? `:` type($clock) `,` type($cond) (`,` qualified(type($substitutions))^)?
  }];
}

def SkipOp : FIRRTLOp<"skip", [Pure]> {
  let summary = "Skip statement";
  let description = [{
    Skip Statement:
    ```
       %firrtl.skip
    ```

    This is a no-op statement.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def StopOp : FIRRTLOp<"stop"> {
  let summary = "Stop Statement";

  let arguments = (ins ClockType:$clock, UInt1Type:$cond, I32Attr:$exitCode,
                       StrAttr:$name);
  let results = (outs);

  let assemblyFormat = "$clock `,` $cond `,` $exitCode `` custom<StopAttrs>(attr-dict) `:` type($clock) `,` type($cond)";
}

class VerifOp<string mnemonic, list<Trait> traits = []> :
    FIRRTLOp<mnemonic, traits> {
  let arguments = (ins
    ClockType:$clock,
    UInt1Type:$predicate,
    UInt1Type:$enable,
    StrAttr:$message,
    Variadic<AnyType>:$substitutions,
    StrAttr:$name,
    DefaultValuedAttr<BoolAttr,"false">:$isConcurrent,
    DefaultValuedAttr<EventControlAttr,"EventControl::AtPosEdge">:$eventControl
  );

  let hasCanonicalizer = true;

  let assemblyFormat = [{
    $clock `,` $predicate `,` $enable `,`
    $message (`(` $substitutions^ `)`)? `:` type($clock) `,` type($predicate) `,` type($enable) (`,` qualified(type($substitutions))^)?
    custom<VerifAttrs>(attr-dict)
  }];
}

def AssertOp : VerifOp<"assert"> {
  let summary = "Assert Verification Statement";
}

def AssumeOp : VerifOp<"assume"> {
  let summary = "Assume Verification Statement";
}

def CoverOp : VerifOp<"cover"> {
  let summary = "Cover Verification Statement";
}

def WhenOp : FIRRTLOp<"when", [SingleBlock, NoTerminator, NoRegionArguments,
                               RecursiveMemoryEffects, RecursivelySpeculatable]> {
  let summary = "When Statement";
  let description = [{
    The "firrtl.when" operation represents a conditional.  Connections within
    a conditional statement that connect to previously declared components hold
    only when the given condition is high. The condition must have a 1-bit
    unsigned integer type.
  }];
  let arguments = (ins UInt1Type:$condition);
  let results = (outs);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$condition, "bool":$withElseRegion,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>
  ];

  let assemblyFormat =
    "$condition `:` type($condition) $thenRegion (`else` $elseRegion^)? attr-dict-with-keyword";

  let extraClassDeclaration = [{

    Block &getThenBlock() {
      assert(!getThenRegion().empty() && "Unexpected empty 'then' region.");
      return getThenRegion().front();
    }

    OpBuilder getThenBodyBuilder() {
      Block &body = getThenBlock();
      return OpBuilder::atBlockEnd(&body);
    }

    bool hasElseRegion() {
      return !getElseRegion().empty();
    }

    void createElseRegion();

    Block &getElseBlock() {
      assert(hasElseRegion() && "Unexpected empty 'else' region.");
      return getElseRegion().front();
    }

    OpBuilder getElseBodyBuilder() {
      Block &body = getElseBlock();
      return OpBuilder::atBlockEnd(&body);
    }
  }];
}

def MatchOp : FIRRTLOp<"match", [SingleBlock, NoTerminator,
    RecursiveMemoryEffects, RecursivelySpeculatable]> {
  let summary = "Match Statement";
  let description = [{
    The "firrtl.match" operation represents a pattern matching statement on a
    enumeration. This operation does not return a value and cannot be used as an
    expression. Last connect semantics work similarly to a when statement.

    Example:
    ```mlir
      firrtl.match %in : !firrtl.enum<Some: uint<1>, None: uint<0>> {
        case Some(%arg0) {
          !firrtl.strictconnect %w, %arg0 : !firrtl.uint<1>
        }
        case None(%arg0) {
          !firrt.strictconnect %w, %c1 : !firrtl.uint<1>
        }
      }
    ```
  }];
  let arguments = (ins FEnumType:$input, I32ArrayAttr:$tags);
  let results = (outs);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input,
                   "::mlir::ArrayAttr":$tags,
                   "::llvm::MutableArrayRef<std::unique_ptr<Region>>":$regions)>
  ];

  let extraClassDeclaration = [{
    IntegerAttr getFieldIndexAttr(size_t caseIndex) {
      return cast<IntegerAttr>(getTags()[caseIndex]);
    }

    uint32_t getFieldIndex(size_t caseIndex) {
      return getFieldIndexAttr(caseIndex).getUInt();
    }
  }];
}

def PropAssignOp : FIRRTLOp<"propassign",
  [FConnectLike, SameTypeOperands, ParentOneOf<["FModuleOp", "ClassOp"]>]> {
  let summary = "Assign to a sink property value.";
  let description = [{
    Assign an output property value. The types must match exactly.

    Example:
    ```mlir
      firrtl.propassign %dest, %src : !firrtl.string
    ```
    }];
  let arguments = (ins PropertyType:$dest, PropertyType:$src);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$dest `,` $src attr-dict `:` qualified(type($dest))";
  let extraClassDeclaration = [{
    static ConnectBehaviorKind getConnectBehaviorKind() {
      return ConnectBehaviorKind::StaticSingleConnect;
    }
  }];
}

def ForceOp : FIRRTLOp<"force", [SameTypeOperands]> {
  let summary = "Force procedural statement";
  let description = "Maps to the corresponding `sv.force` operation.";
  let arguments = (ins FIRRTLBaseType:$dest, FIRRTLBaseType:$src);
  let results = (outs);
  let assemblyFormat =
    "$dest `,` $src attr-dict `:` qualified(type($dest)) `,` qualified(type($src))";
}

//===- Reference force/release --------------------------------------------===//

class ForceRefTypeConstraint<string ref, string base>
  : TypesMatchWith<"reference type of " # ref # " should be RWProbe of " # base,
                   base, ref,
                   "RefType::get(firrtl::type_cast<FIRRTLBaseType>($_self).getAllConstDroppedType(), true)">;

def RefForceOp : FIRRTLOp<"ref.force",[ForceRefTypeConstraint<"dest", "src">]> {
  let summary = "FIRRTL force statement";
  let description = "Force a RWProbe to the specified value using the specified clock and predicate.";
  let arguments = (ins ClockType:$clock, UInt1Type:$predicate, RWProbe:$dest, FIRRTLBaseType:$src);
  let assemblyFormat =
    "$clock `,` $predicate `,` $dest `,` $src attr-dict `:` type($clock) `,` type($predicate) `,` qualified(type($src))";
  let hasCanonicalizer = 1;
}
def RefForceInitialOp : FIRRTLOp<"ref.force_initial",[ForceRefTypeConstraint<"dest", "src">]> {
  let summary = "FIRRTL force_initial statement";
  let description = "Force a RWProbe to the specified value continuously.";
  let arguments = (ins UInt1Type:$predicate, RWProbe:$dest, FIRRTLBaseType:$src);
  let assemblyFormat =
    "$predicate `,` $dest `,` $src attr-dict `:` type($predicate) `,` qualified(type($src))";
  let hasCanonicalizer = 1;
}
def RefReleaseOp : FIRRTLOp<"ref.release"> {
  let summary = "FIRRTL release statement";
  let description = "Release the target RWProbe using the specified clock and predicate.";
  let arguments = (ins ClockType:$clock, UInt1Type:$predicate, RWProbe:$dest);
  let assemblyFormat =
    "$clock `,` $predicate `,` $dest attr-dict `:` type($clock) `,` type($predicate) `,` qualified(type($dest))";
  let hasCanonicalizer = 1;
}
def RefReleaseInitialOp : FIRRTLOp<"ref.release_initial"> {
  let summary = "FIRRTL release_initial statement";
  let description = "Release the target RWProbe.";
  let arguments = (ins UInt1Type:$predicate, RWProbe:$dest);
  let assemblyFormat = "$predicate `,` $dest attr-dict `:` type($predicate) `,` qualified(type($dest))";
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Verif Dialect Intrinsics
//===----------------------------------------------------------------------===//

class VerifIntrinsicOp<string mnemonic, list<Trait> traits = []> :
    FIRRTLOp<"int.verif." # mnemonic, traits> {
  let summary = "FIRRTL variant of `verif." # mnemonic # "`";
  let description = "See `verif." # mnemonic # "` op in the Verif dialect.";
  let arguments = (ins UInt1Type:$property, OptionalAttr<StrAttr>:$label);
  let assemblyFormat = [{
    operands attr-dict `:` type(operands)
  }];
}

def VerifAssertIntrinsicOp : VerifIntrinsicOp<"assert">;
def VerifAssumeIntrinsicOp : VerifIntrinsicOp<"assume">;
def VerifCoverIntrinsicOp : VerifIntrinsicOp<"cover">;

//===----------------------------------------------------------------------===//
// Group Op
//===----------------------------------------------------------------------===//

def GroupOp : FIRRTLOp<
  "group",
  [SingleBlock, NoTerminator, NoRegionArguments,
   ParentOneOf<["firrtl::FModuleOp", "firrtl::GroupOp"]>,
   DeclareOpInterfaceMethods<SymbolUserOpInterface>]
> {
  let summary = "A definition of an optional group";
  let description = [{
    The `firrtl.group` operation defines optional code that is conditionally
    part of a `firrtl.module`.  This is typically used to store verification or
    debugging code that is lowered to a SystemVerilog `bind` instantiation.  An
    optional group can read from (capture) values defined in parent group
    definitions or in the parent module, but may not write to hardware declared
    outside the group.

    A `firrtl.group` must refer to an existing optional group declaration
    (`firrtl.declgroup`) via a symbol reference.  A nested `firrtl.group` refers
    to a nested group declaration via a nested symbol reference.
  }];
  let arguments = (ins SymbolRefAttr:$groupName);
  let results = (outs);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    $groupName $region attr-dict
  }];
  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLSTATEMENTS_TD
