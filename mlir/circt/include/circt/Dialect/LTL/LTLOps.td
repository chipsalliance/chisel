//===- LTLOps.td - LTL dialect operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLOPS_TD
#define CIRCT_DIALECT_LTL_LTLOPS_TD

include "circt/Dialect/LTL/LTLDialect.td"
include "circt/Dialect/LTL/LTLTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LTLOp<string mnemonic, list<Trait> traits = []> :
  Op<LTLDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Generic Operations
//===----------------------------------------------------------------------===//

class AssocLTLOp<string mnemonic, list<Trait> traits = []> :
  LTLOp<mnemonic, traits # [Pure, Commutative, InferTypeOpInterface,
                            DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<LTLAnyPropertyType>:$inputs);
  let results = (outs LTLAnyPropertyType:$result);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];
}

def AndOp : AssocLTLOp<"and"> {
  let summary = "A conjunction of booleans, sequences, or properties.";
  let description = [{
    If any of the `$inputs` is of type `!ltl.property`, the result of the op is
    an `!ltl.property`. Otherwise it is an `!ltl.sequence`.
  }];
}

def OrOp : AssocLTLOp<"or"> {
  let summary = "A disjunction of booleans, sequences, or properties.";
  let description = [{
    If any of the `$inputs` is of type `!ltl.property`, the result of the op is
    an `!ltl.property`. Otherwise it is an `!ltl.sequence`.
  }];
}

//===----------------------------------------------------------------------===//
// Sequences
//===----------------------------------------------------------------------===//

def DelayOp : LTLOp<"delay", [Pure]> {
  let arguments = (ins
    LTLAnySequenceType:$input,
    I64Attr:$delay,
    OptionalAttr<I64Attr>:$length);
  let results = (outs LTLSequenceType:$result);
  let assemblyFormat = [{
    $input `,` $delay (`,` $length^)? attr-dict `:` type($input)
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let summary = "Delay a sequence by a number of cycles.";
  let description = [{
    Delays the `$input` sequence by the number of cycles specified by `$delay`.
    The delay must be greater than or equal to zero. The optional `$length`
    specifies during how many cycles after the initial delay the sequence can
    match. Omitting `$length` indicates an unbounded but finite delay. For
    example:

    - `ltl.delay %seq, 2, 0` delays `%seq` by exactly 2 cycles. The resulting
      sequence matches if `%seq` matches exactly 2 cycles in the future.
    - `ltl.delay %seq, 2, 2` delays `%seq` by 2, 3, or 4 cycles. The resulting
      sequence matches if `%seq` matches 2, 3, or 4 cycles in the future.
    - `ltl.delay %seq, 2` delays `%seq` by 2 or more cycles. The number of
      cycles is unbounded but finite, which means that `%seq` *has* to match at
      some point, instead of effectively never occuring by being delayed an
      infinite number of cycles.
    - `ltl.delay %seq, 0, 0` is equivalent to just `%seq`.

    #### Clocking

    The cycle delay specified on the operation refers to a clocking event. This
    event is not directly specified by the delay operation itself. Instead, the
    [`ltl.clock`](#ltlclock-circtltlclockop) operation can be used to associate
    all delays within a sequence with a clock.
  }];
}

def ConcatOp : LTLOp<"concat", [Pure]> {
  let arguments = (ins Variadic<LTLAnySequenceType>:$inputs);
  let results = (outs LTLSequenceType:$result);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let summary = "Concatenate sequences into a longer sequence.";
  let description = [{
    Concatenates all of the `$inputs` sequences one after another into one
    longer sequence. The sequences are arranged such that the end time of the
    previous sequences coincides with the start time of the next sequence. This
    means there is no implicit cycle of delay between the concatenated
    sequences, which may be counterintuitive.

    If a sequence should follow in the cycle after another sequence finishes,
    that cycle of delay needs to be explicit. For example, *"u followed by v in
    next cycle"* (`u ##1 v` in SVA) is represented as
    `concat(u, delay(v, 1, 0))`:
    ```
    %0 = ltl.delay %v, 1, 0 : i1
    ltl.concat %u, %v : !ltl.sequence, !ltl.sequence
    ```
    The resulting sequence checks for `u` in the first cycle and `v` in the
    second, `[u, v]` in short.

    Without this explicit delay, the previous sequence's end overlaps with the
    next sequence's start. For example, consider the two sequences `u = a ##1 b`
    and `v = c ##1 d`, which check for `a` and `c` in the first, and `b` and `d`
    in the second cycle. When these two sequences are concatenated,
    `concat(u, v)`, the end time of the first sequence coincides with the start
    time of the second. As a result, the check for `b` at the end of the first
    sequence will coincide with the check for `c` at the start of the second
    sequence: `concat(u, v) = a ##1 (b && c) ##1 d`. The resulting sequence
    checks for `a` in the first cycle, `b` and `c` in the second, and `d` in the
    third, `[a, (b && c), d]` in short.

    By making the delay between concatenated sequences explicit, the `concat`
    operation behaves nicely in the presence of zero-length sequences. An empty,
    zero-length sequence in a concatenation behaves as if the sequence wasn't
    present at all. Compare this to SVAs which struggle with empty sequences.
    For example, `x ##1 y ##1 z` would become `x ##2 z` if `y` was empty.
    Similarly, expressing zero or more repetitions of a sequence, `w ##[*]`, is
    challenging in SVA since concatenation always implies a cycle of delay, but
    trivial if the delay is made explicit. This is related to the handling of
    empty rules in a parser's grammar.

    Note that concatenating two boolean values *a* and *b* is equivalent to
    computing the logical AND of them. Booleans are sequences that check if the
    boolean is true in the current cycle, which means that the sequence starts
    and ends in the same cycle. Since concatenation aligns the sequences such
    that end time of *a* and start time of *b* coincide, the resulting sequence
    checks if *a* and *b* both are true in the current cycle, which is an AND
    operation.
  }];
}

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

def NotOp : LTLOp<"not", [Pure]> {
  let arguments = (ins LTLAnyPropertyType:$input);
  let results = (outs LTLPropertyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];

  let summary = "A negation of a property.";
  let description = [{
    Negates the `$input` property. The resulting property evaluates to true if
    `$input` evaluates to false, and it evaluates to false if `$input` evaluates
    to true.
  }];
}

def ImplicationOp : LTLOp<"implication", [Pure]> {
  let arguments = (ins LTLAnySequenceType:$antecedent,
                       LTLAnyPropertyType:$consequent);
  let results = (outs LTLPropertyType:$result);
  let assemblyFormat = [{
    operands attr-dict `:` type(operands)
  }];

  let summary = "Only check a property after a sequence matched.";
  let description = [{
    Preconditions the checking of the `$consequent` property on the
    `$antecedent` sequence. In a nutshell, if the `$antecedent` sequence matches
    at a given point in time, the `$consequent` property is checked starting at
    the point in time at which the matched sequence ends. The result property of
    the `ltl.implication` holds if the `$consequent` holds. Conversely, if the
    `$antecedent` does *not* match at a given point in time, the result property
    trivially holds. This is conceptually identical to the implication operator
    →, but with additional temporal semantics.
  }];
}

def EventuallyOp : LTLOp<"eventually", [Pure]> {
  let arguments = (ins LTLAnyPropertyType:$input);
  let results = (outs LTLPropertyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];

  let summary = "Ensure that a property will hold at some time in the future.";
  let description = [{
    Checks that the `$input` property will hold at a future time. This operator
    is strong: it requires that the `$input` holds after a *finite* number of
    cycles. The operator does *not* hold if the `$input` can't hold in the
    future.
  }];
}

//===----------------------------------------------------------------------===//
// Clocking
//===----------------------------------------------------------------------===//

// Edge behavior enum for always block.  See SV Spec 9.4.2.

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"Pos", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"Neg", 1, "negedge">;
/// AtEdge is syntactic sugar for AtPosEdge or AtNegEdge.
def AtEdge   : I32EnumAttrCase<"Both", 2, "edge">;

def ClockEdgeAttr : I32EnumAttr<"ClockEdge", "clock edge",
                                [AtPosEdge, AtNegEdge, AtEdge]> {
  let cppNamespace = "circt::ltl";
}

def ClockOp : LTLOp<"clock", [
  Pure, InferTypeOpInterface, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let arguments = (ins LTLAnyPropertyType:$input, ClockEdgeAttr:$edge, I1:$clock);
  let results = (outs LTLSequenceOrPropertyType:$result);
  let assemblyFormat = [{
    $input `,` $edge $clock attr-dict `:` type($input)
  }];

  let summary = "Specify the clock for a property or sequence.";
  let description = [{
    Specifies the `$edge` on a given `$clock` to be the clock for an `$input`
    property or sequence. All cycle delays in the `$input` implicitly refer to a
    clock that advances the state to the next cycle. The `ltl.clock` operation
    provides that clock. The clock applies to the entire property or sequence
    expression tree below `$input`, up to any other nested `ltl.clock`
    operations.

    The operation returns a property if the `$input` is a property, and a
    sequence otherwise.
  }];
}

def DisableOp : LTLOp<"disable", [Pure]> {
  let arguments = (ins LTLAnyPropertyType:$input, I1:$condition);
  let results = (outs LTLPropertyType:$result);
  let assemblyFormat = [{
    $input `if` $condition attr-dict `:` type($input)
  }];
  let hasFolder = 1;

  let summary = "Disable the evaluation of a property based on a condition.";
  let description = [{
    Creates a new property which evaluates the given `$input` property only if
    the given disable `$condition` is false throughout the entire evaluation. If
    the `$condition` is true at any point in time during the evaluation of
    `$input`, the resulting property is disabled.

    The disabling is "infectious". If a property is disabled, it also implicitly
    disables all properties that use it. Consider the following example:
    ```
    %0 = ltl.disable %prop if %cond
    %1 = ltl.or %0, %otherProp
    ```
    If the property `%0` is disabled, the derived property `%1` is also
    disabled.

    As a result, it is always legal to canonicalize the IR into a form where
    there is only one `ltl.disable` operation at the root of a property
    expression.

    Note that a property being disabled based on a condition is different from a
    property that trivially evaluates to true if the condition does not hold.
    The former ensures that a property is only checked when a certain condition
    is true, but the number of cases in which the property holds or doesn't hold
    remains unchanged. The latter adds additional cases where the property
    holds, which can offer a solver unintended ways to make assertions or
    coverage proofs derived from the property pass. For example:

    ```
    %p0 = ltl.or %prop, %cond
    %p1 = ltl.disable %prop if %cond
    ```

    `$cond` being true would disable the evaluation of `%p0` and would make
    `%p1` evaluate to true. These are subtly different. If used in an  assertion
    during simulation, `$cond` would adequately disable triggering of the
    assertion in both cases. However, if used in a formal verification setting
    where proofs for `%p0` and `%p1` always holding or never holding are sought,
    a solver might use `%cond = true` to trivially make `%p0` hold, which is not
    possible for `%p1`. Consider the following more concrete example:

    ```
    %p2 = ltl.or %protocolCorrectProperty, %reset
    %p3 = ltl.disable %protocolCorrectProperty if %reset
    verif.cover %p2
    verif.cover %p3
    ```

    The intent is to formally prove coverage for `%protocolCorrectProperty`
    while the circuit is in regular operation (i.e., out of reset). A formal
    solver would trivially prove coverage for `%p2` by assigning
    `%reset = true`, but would have to actually prove coverage for the
    underlying `%protocolCorrectProperty` for `%p3`. The latter is almost always
    the intended behavior.
  }];
}

#endif // CIRCT_DIALECT_LTL_LTLOPS_TD
