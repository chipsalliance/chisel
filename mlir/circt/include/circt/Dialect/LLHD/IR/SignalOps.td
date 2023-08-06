//===- SignalOps.td - LLHD signal operations ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for LLHD signal creation and manipulation.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

def LLHD_SigOp : LLHD_Op<"sig", [
    ParentOneOf<["llhd::EntityOp", "llhd::ProcOp"]>,
    TypesMatchWith<
      "type of 'init' and underlying type of 'signal' have to match.",
      "init", "result", "SigType::get($_self)">
  ]> {
  let summary = "Create a signal.";
  let description = [{
    The `llhd.sig` instruction introduces a new signal in the IR. The input
    operand determines the initial value carried by the signal, while the
    result type will always be a signal carrying the type of the init operand.
    A signal defines a unique name within the entity it resides in.

    Syntax:

    ```
    sig-op ::= ssa-id `=` `llhd.sig` sig-name ssa-init attr-dict `:` init-type
    ```

    Example:

    ```mlir
    %init_i64 = llhd.const 123 : i64
    %sig_i64 = llhd.sig "foo" %init_64 : i64

    %init_i1 = llhd.const 1 : i1
    %sig_i1 = llhd.sig "bar" %init_i1 : i1
    ```

    The first `llhd.sig` instruction creates a new signal named "foo", carrying
    an `i64` type with initial value of 123, while the second one creates a new
    signal named "bar", carrying an `i1` type with initial value of 1.
  }];

  let arguments = (ins StrAttr: $name, LLHD_AnyUnderlyingType: $init);
  let results = (outs LLHD_AnySigType: $result);

  let assemblyFormat = "$name $init attr-dict `:` qualified(type($init))";
}

def LLHD_PrbOp : LLHD_Op<"prb", [
    TypesMatchWith<
      "type of 'result' and underlying type of 'signal' have to match.",
      "signal", "result", "$_self.cast<SigType>().getUnderlyingType()">
  ]> {
  let summary = "Probe a signal.";
  let description = [{
    The `llhd.prb` instruction probes a signal and returns the value it
    currently carries as a new SSA operand. The result type is always
    the type carried by the signal.

    Syntax:

    ```
    prb-op ::= ssa-id `=` `llhd.prb` ssa-sig attr-dict `:` !llhd.sig<type>
    ```

    Example:

    ```mlir
    %const_i1 = llhd.const 1 : i1
    %sig_i1 = llhd.sig %const_i1 : i1
    %prbd = llhd.prb %sig_i1 : !llhd.sig<i1>
    ```
  }];

  let arguments = (ins Arg<LLHD_AnySigType, "the signal to probe from",
                           [MemRead]>: $signal);
  let results = (outs LLHD_AnyUnderlyingType: $result);

  let assemblyFormat = "$signal attr-dict `:` qualified(type($signal))";
}

def LLHD_OutputOp : LLHD_Op<"output", [
    TypesMatchWith<
      "type of 'value' and underlying type of 'result' have to match.",
      "value", "result", "SigType::get($_self)">
  ]> {
  let summary = "Introduce a new signal and drive a value onto it.";
  let description = [{
    The `llhd.output` operation introduces a new signal and continuously
    drives a the given value onto it after a given time-delay. The same
    value is used to initialize the signal in the same way as the 'init'
    value in `llhd.sig`. An optional name can be given to the created signal.
    This shows up, e.g., in the simulation trace.

    Example:

    ```mlir
    %value = llhd.const 1 : i1
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    %sig = llhd.output "sigName" %value after %time : i1

    // is equivalent to

    %value = llhd.const 1 : i1
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    %sig = llhd.sig "sigName" %value : i1
    llhd.drv %sig, %value after %time : !llhd.sig<i1>
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>: $name,
                       LLHD_AnyUnderlyingType: $value,
                       LLHD_TimeType: $time);

  let results = (outs LLHD_AnySigType: $result);

  let assemblyFormat = [{
    ( $name^ )? $value `after` $time attr-dict `:` qualified(type($value))
  }];
}

def LLHD_DrvOp : LLHD_Op<"drv", [
    TypesMatchWith<
      "type of 'value' and underlying type of 'signal' have to match.",
      "signal", "value", "$_self.cast<SigType>().getUnderlyingType()">
  ]> {
  let summary = "Drive a value into a signal.";
  let description = [{
    The `llhd.drv` operation drives a new value onto a signal. A time
    operand also has to be passed, which specifies the frequency at which
    the drive will be performed. An optional enable value can be passed as
    last argument. In this case the drive will only be performed if the
    value is 1. In case no enable signal is passed the drive will always be
    performed. This operation does not define any new SSA operands.

    Syntax:

    ```
    drv-op ::= `llhd.drv` ssa-signal `,` ssa-const `after` ssa-time
      (`if` ssa-enable)? `:` !llhd.sig<const-type>
    ```

    Example:

    ```mlir
    %init = llhd.const 1 : i1
    %en = llhd.const 0 : i1
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    %sig = llhd.sig %init : i1
    %new = llhd.not %init : i1

    llhd.drv %sig, %new after %time : !llhd.sig<i1>
    llhd.drv %sig, %new after %time if %en : !llhd.sig<i1>
    ```
  }];

  let arguments = (ins Arg<LLHD_AnySigType, "the signal to drive to",
                           [MemWrite]>: $signal,
                       LLHD_AnyUnderlyingType: $value,
                       LLHD_TimeType: $time,
                       Optional<I1>: $enable);

  let assemblyFormat = [{
    $signal `,` $value `after` $time ( `if` $enable^ )? attr-dict `:`
    qualified(type($signal))
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def REG_MODE_LOW  : I64EnumAttrCase<"low", 0>;
def REG_MODE_HIGH : I64EnumAttrCase<"high", 1>;
def REG_MODE_RISE : I64EnumAttrCase<"rise", 2>;
def REG_MODE_FALL : I64EnumAttrCase<"fall", 3>;
def REG_MODE_BOTH : I64EnumAttrCase<"both", 4>;

def LLHD_RegModeAttr : I64EnumAttr<"RegMode", "", [
    REG_MODE_LOW, REG_MODE_HIGH, REG_MODE_RISE, REG_MODE_FALL, REG_MODE_BOTH
  ]> {
  let cppNamespace = "::circt::llhd";
}

def LLHD_RegModeArrayAttr
    : TypedArrayAttrBase<LLHD_RegModeAttr, "reg mode array attribute"> {}

def LLHD_RegOp : LLHD_Op<"reg", [
    HasParent<"EntityOp">,
    AttrSizedOperandSegments
  ]> {
  let summary = "Represents a storage element";
  let description = [{
    This instruction represents a storage element. It drives its output onto
    the 'signal' value. An arbitrary amount of triggers can be added to the
    storage element. However, at least one is required. They are quadruples
    consisting of the new value to be stored if the trigger applies, the
    mode and trigger value which specify when this trigger has to be applied
    as well as a delay. Optionally, each triple may also have a gate
    condition, in this case the trigger only applies if the gate is one. If
    multiple triggers apply the left-most in the list takes precedence.

    There are five modes available:

    | Mode   | Meaning                                                         |
    |--------|-----------------------------------------------------------------|
    | "low"  | Storage element stores `value` while the `trigger` is low. Models active-low resets and low-transparent latches.
    | "high" | Storage element stores `value` while the `trigger` is high. Models active-high resets and high-transparent latches.
    | "rise" | Storage element stores `value` upon the rising edge of the `trigger`. Models rising-edge flip-flops.
    | "fall" | Storage element stores `value` upon the falling edge of the `trigger`. Models falling-edge flip-flops.
    | "both" | Storage element stores `value` upon the a rising or a falling edge of the `trigger`. Models dual-edge flip-flops.

    This instruction may only be used in an LLHD entity.

    Syntax:

    ```
    reg-op ::= `llhd.reg` signal-ssa-value
      ( `,` `(` value-ssa-value `,` mode-string trigger-ssa-value `after`
      delay-ssa-value ( `if` gate-ssa-value )? `:` value-type )+
      attr-dict `:` signal-type
    ```

    Examples:

    A rising, falling, and dual-edge triggered flip-flop:

    ```mlir
    llhd.reg %Q, (%D, "rise" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
    llhd.reg %Q, (%D, "fall" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
    llhd.reg %Q, (%D, "both" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
    ```

    A rising-edge triggered flip-flop with active-low reset:

    ```mlir
    llhd.reg %Q, (%init, "low" %RSTB after %T : !llhd.sig<i8>),
      (%D, "rise" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
    ```

    A rising-edge triggered enable flip-flop with active-low reset:

    ```mlir
    llhd.reg %Q, (%init, "low" %RSTB after %T : !llhd.sig<i8>),
      (%D, "rise" %CLK after %T if %EN : !llhd.sig<i8>) : !llhd.sig<i8>
    ```

    A transparent-low and transparent-high latch:

    ```mlir
    llhd.reg %Q, (%D, "low" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
    llhd.reg %Q, (%D, "high" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
    ```

    An SR latch:

    ```mlir
    %0 = llhd.const 0 : i1
    %1 = llhd.const 1 : i1
    llhd.reg %Q, (%0, "high" %R after %T : !llhd.sig<i1>),
      (%1, "high" %S after %T : !llhd.sig<i1>) : !llhd.sig<i1>
    ```
  }];

  let arguments = (ins
    LLHD_AnySigType: $signal,
    LLHD_RegModeArrayAttr: $modes,
    Variadic<AnyTypeOf<[LLHD_AnyUnderlyingType, LLHD_AnySigType]>>: $values,
    Variadic<I1>: $triggers,
    Variadic<LLHD_TimeType>: $delays,
    Variadic<I1>: $gates,
    I64ArrayAttr: $gateMask);

  let extraClassDeclaration = [{
    static StringRef getModeAttrName() { return "modes"; }
    static RegMode getRegModeByName(StringRef name) {
      std::optional<RegMode> optional =  symbolizeRegMode(name);
      assert(optional && "Invalid RegMode string.");
      return *optional;
    }

    bool hasGate(unsigned index) {
      assert(index < getGateMask().getValue().size() && "Index out of range.");
      return getGateMask().getValue()[index].cast<IntegerAttr>().getInt() != 0;
    }

    Value getGateAt(unsigned index) {
      assert(index < getGateMask().getValue().size() && "Index out of range.");
      if (!hasGate(index)) return Value();
      return
        getGates()[getGateMask().getValue()[index].cast<IntegerAttr>().getInt()-1];
    }

    RegMode getRegModeAt(unsigned index) {
      assert(index < getModes().getValue().size() && "Index out of range.");
      return (RegMode)getModes().getValue()[index].cast<IntegerAttr>().getInt();
    }
  }];

  let hasVerifier = 1;
}
