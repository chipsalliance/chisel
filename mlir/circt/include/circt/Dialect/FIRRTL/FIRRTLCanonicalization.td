//===- FIRRTLCanonicalization.td - FIRRTL canonicalization -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The canonicalization patterns for the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD

include "FIRRTLDeclarations.td"
include "FIRRTLExpressions.td"
include "FIRRTLStatements.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

/// Constraint that matches a ConstantOp, SpecialConstantOp, or AggregateConstantOp.
def AnyConstantOp : Constraint<CPred<[{
  isa_and_nonnull<ConstantOp, SpecialConstantOp, AggregateConstantOp>
    ($0.getDefiningOp())
  }]>>;

/// Constraint that matches non-constant operations. Used to ensure that the
/// const-on-LHS rewriting patterns converge in case both operands are constant.
def NonConstantOp : Constraint<CPred<[{
  !isa_and_nonnull<ConstantOp, SpecialConstantOp, AggregateConstantOp>
    ($0.getDefiningOp())
  }]>>;

// Constraint that matches empty attribute dictionaries.  Used to ensure that
// there are no FIRRTL annotation on an operation.
def EmptyAttrDict : Constraint<CPred<"$0.empty()">>;

def GetDroppableNameAttr : NativeCodeCall<
  "NameKindEnumAttr::get($_builder.getContext(), NameKindEnum::DroppableName) ">;
def NonEmptyAttr : Constraint<CPred<"!$0.getValue().empty()">>;
def NullAttr : Constraint<CPred<"!$0">>;

// Constraint that enforces equal types
def EqualTypes : Constraint<CPred<"areAnonymousTypesEquivalent($0.getType(), $1.getType())">>;

// Constraint that enforces equal type sizes
def EqualIntSize : Constraint<CPred<"type_cast<IntType>($0.getType()).getWidth() == type_cast<IntType>($1.getType()).getWidth()">>;

// sizeof(0) >= sizeof(1)
def IntTypeWidthGEQ32 : Constraint<CPred<
  "type_cast<IntType>($0.getType()).getBitWidthOrSentinel() >= type_cast<IntType>($1.getType()).getBitWidthOrSentinel()">>;

// sizeof(0) > sizeof(1)
def IntTypeWidthGT32 : Constraint<CPred<
  "type_cast<IntType>($0.getType()).getBitWidthOrSentinel() > type_cast<IntType>($1.getType()).getBitWidthOrSentinel()">>;

// Constraint that enforces int types
def IntTypes : Constraint<CPred<"type_isa<IntType>($0.getType())">>;

// Constraint that enforces sint types
def SIntTypes : Constraint<CPred<"type_isa<SIntType>($0.getType())">>;

// Constraint that enforces uint types
def UIntTypes : Constraint<CPred<"type_isa<UIntType>($0.getType())">>;

// Constraint that enforces types mismatches
def mismatchTypes : Constraint<CPred<"!areAnonymousTypesEquivalent($0.getType(), $1.getType())">>;

// Constraint that enforces types of known width
def KnownWidth : Constraint<CPred<[{
  type_isa<FIRRTLBaseType>($0.getType()) &&
    !type_cast<FIRRTLBaseType>($0.getType()).getRecursiveTypeProperties().hasUninferredWidth
  }]>>;

/// Constraint that matches a false SpecialConstantOp
def FalseSpecialConstantOp : Constraint<CPred<
  "$0.getDefiningOp<SpecialConstantOp>() &&"
  "$0.getDefiningOp<SpecialConstantOp>().getValue() == false">>;

/// Constraint that matches a true SpecialConstantOp
def TrueSpecialConstantOp : Constraint<CPred<
  "$0.getDefiningOp<SpecialConstantOp>() &&"
  "$0.getDefiningOp<SpecialConstantOp>().getValue() == true">>;

/// Constraint that matches a zero ConstantOp or SpecialConstantOp.
def ZeroConstantOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<ConstantOp>() &&"
        "$0.getDefiningOp<ConstantOp>().getValue().isZero()">,
  FalseSpecialConstantOp.predicate
]>>;

/// Constraint that matches a one ConstantOp or SpecialConstantOp.
def OneConstantOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<ConstantOp>() &&"
        "$0.getDefiningOp<ConstantOp>().getValue().isOne()">,
  TrueSpecialConstantOp.predicate
]>>;

/// Constraint that matches a not-all-ones ConstantOp or SpecialConstantOp.
def NotAllOneConstantOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<ConstantOp>() &&"
        "!$0.getDefiningOp<ConstantOp>().getValue().isAllOnes()">,
  FalseSpecialConstantOp.predicate
]>>;

/// Constraint that matches an all ones ConstantOp.
def AllOneConstantOp : Constraint<CPred<"$0.getDefiningOp<ConstantOp>() && $0.getDefiningOp<ConstantOp>().getValue().isAllOnes()">>;

// Delta between type widths
def TypeWidthAdjust32 : NativeCodeCall<
  "$_builder.getI32IntegerAttr(type_cast<FIRRTLBaseType>($0.getType()).getBitWidthOrSentinel() - type_cast<FIRRTLBaseType>($1.getType()).getBitWidthOrSentinel())">;

/// Drop the writer to the first argument and passthrough the second
def DropWrite : NativeCodeCall<"dropWrite($_builder, $0, $1)">;

def MoveNameHint : NativeCodeCall<"moveNameHint($0, $1)">;

def GetEmptyString : NativeCodeCall<
  "StringAttr::get($_builder.getContext(), {}) ">;

// leq(const, x) -> geq(x, const)
def LEQWithConstLHS : Pat<
  (LEQPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (GEQPrimOp $rhs, $lhs)),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// lt(const, x) -> gt(x, const)
def LTWithConstLHS : Pat<
  (LTPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (GTPrimOp $rhs, $lhs)),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// geq(const, x) -> leq(x, const)
def GEQWithConstLHS : Pat<
  (GEQPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (LEQPrimOp $rhs, $lhs)),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// gt(const, x) -> lt(x, const)
def GTWithConstLHS : Pat<
  (GTPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (LTPrimOp $rhs, $lhs)),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// not(not(x)) -> x
def NotNot : Pat <
  (NotPrimOp:$old (NotPrimOp $x)),
  (MoveNameHint $old, (AsUIntPrimOp $x)),
  []>;

// connect(a:t,b:t) -> strictconnect(a,b)
def ConnectSameType : Pat<
  (ConnectOp $dst, $src),
  (StrictConnectOp $dst, $src),
  [(EqualTypes $dst, $src), (KnownWidth $dst)]>;

// connect(a:t1,b:(t2<t1) -> strictconnect(a,(pad b))
def ConnectExtension : Pat<
  (ConnectOp $dst, $src),
  (StrictConnectOp $dst, (PadPrimOp $src,
    (NativeCodeCall<"type_cast<FIRRTLBaseType>($0.getType()).getBitWidthOrSentinel()"> $dst))),
  [(IntType $dst), (IntType $src), (KnownWidth $dst), (KnownWidth $src), (mismatchTypes $src, $dst)]>;

def LimitConstant32 : NativeCodeCall<
  "$_builder.getI32IntegerAttr($0.getValue().getLimitedValue(1ULL << 31))">;

def TypeWidth32 : NativeCodeCall<
  "$_builder.getI32IntegerAttr(type_cast<FIRRTLBaseType>($0.getType()).getBitWidthOrSentinel())">;

// dshl(a, const) -> shl(a, const)
def DShlOfConstant : Pat<
  (DShlPrimOp:$old $x, (ConstantOp $cst)),
  (PadPrimOp (MoveNameHint $old, (ShlPrimOp $x, (LimitConstant32 $cst))), (TypeWidth32 $old)),
  [(KnownWidth $x)]>;

// dshr(a, const) -> shr(a, const)
def DShrOfConstant : Pat<
  (DShrPrimOp:$old $x, (ConstantOp $cst)),
  (PadPrimOp (MoveNameHint $old, (ShrPrimOp $x, (LimitConstant32 $cst))), (TypeWidth32 $old)),
  [(KnownWidth $x)]>;

////////////////////////////////////////////////////////////////////////////////
// Extend arguments to make types match
////////////////////////////////////////////////////////////////////////////////

def extendAnd : Pat <
  (AndPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (AndPrimOp (PadPrimOp $lhs, (TypeWidth32 $old)),
                                 (PadPrimOp $rhs, (TypeWidth32 $old)))),
  [(KnownWidth $lhs), (KnownWidth $rhs), (mismatchTypes $lhs, $rhs)]>;

def extendOr : Pat <
  (OrPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (OrPrimOp (PadPrimOp $lhs, (TypeWidth32 $old)),
                                (PadPrimOp $rhs, (TypeWidth32 $old)))),
  [(KnownWidth $lhs), (KnownWidth $rhs), (mismatchTypes $lhs, $rhs)]>;

def extendXor : Pat <
  (XorPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (XorPrimOp (PadPrimOp $lhs, (TypeWidth32 $old)),
                                 (PadPrimOp $rhs, (TypeWidth32 $old)))),
  [(KnownWidth $lhs), (KnownWidth $rhs), (mismatchTypes $lhs, $rhs)]>;

////////////////////////////////////////////////////////////////////////////////
// Move constants to the right
////////////////////////////////////////////////////////////////////////////////

def moveConstAdd : Pat <
  (AddPrimOp:$old $x, $y),
  (MoveNameHint $old, (AddPrimOp $y, $x)),
  [(AnyConstantOp $x), (NonConstantOp $y)]>;

def moveConstAnd : Pat <
  (AndPrimOp:$old $x, $y),
  (MoveNameHint $old, (AndPrimOp $y, $x)),
  [(AnyConstantOp $x), (NonConstantOp $y)]>;

def moveConstOr : Pat <
  (OrPrimOp:$old $x, $y),
  (MoveNameHint $old, (OrPrimOp $y, $x)),
  [(AnyConstantOp $x), (NonConstantOp $y)]>;

def moveConstXor : Pat <
  (XorPrimOp:$old $x, $y),
  (MoveNameHint $old, (XorPrimOp $y, $x)),
  [(AnyConstantOp $x), (NonConstantOp $y)]>;

////////////////////////////////////////////////////////////////////////////////
// Binary ops
////////////////////////////////////////////////////////////////////////////////

// add(a, 0) -> a
def AddOfZero : Pat <
  (AddPrimOp:$old $x, (ConstantOp:$zcst $_)),
  (PadPrimOp $x, (TypeWidth32 $old)),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

// add(a, a) -> a << 2
def AddOfSelf : Pat <
  (AddPrimOp:$old $x, $x),
  (MoveNameHint $old, (ShlPrimOp $x, (NativeCodeCall<"1">))),
  [(KnownWidth $x)]>;

// add((pad a, n), b) -> pad(add(a, b), n)
def AddOfPad : Pat <
  (AddPrimOp:$old (either (PadPrimOp $a, $m), $b)),
  (PadPrimOp (MoveNameHint $old, (AddPrimOp $a, $b)), (TypeWidth32 $old)),
  [(KnownWidth $a), (KnownWidth $b)]>;

// sub(a, 0) -> a
def SubOfZero : Pat <
  (SubPrimOp:$old $x, (ConstantOp:$zcst $cst)),
  (MoveNameHint $old, (PadPrimOp $x, (TypeWidth32 $old))),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

// sub(0, a) -> -a
def SubFromZeroSigned : Pat <
  (SubPrimOp:$old (ConstantOp:$zcst $cst), $x),
  (PadPrimOp (MoveNameHint $old, (NegPrimOp $x)), (TypeWidth32 $old)),
  [(KnownWidth $x), (ZeroConstantOp $zcst), (SIntType $x)]>;

//sub(0, a) -> -a
def SubFromZeroUnsigned : Pat <
  (SubPrimOp:$old (ConstantOp:$zcst $cst), $x),
  (AsUIntPrimOp (PadPrimOp (MoveNameHint $old, (NegPrimOp $x)), (TypeWidth32 $old))),
  [(KnownWidth $x), (ZeroConstantOp $zcst), (UIntType $x)]>;

//sub(a,a) -> 0
def SubOfSelf : Pat <
  (SubPrimOp:$old $x, $x),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(KnownWidth $x)]>;

// sub((pad a, n), b) -> pad(sub(a, b), n)
def SubOfPadL : Pat <
  (SubPrimOp:$old (PadPrimOp $a, $m), $b),
  (MoveNameHint $old, (PadPrimOp (SubPrimOp $a, $b), (TypeWidth32 $old))),
  [(KnownWidth $a), (KnownWidth $b)]>;

// sub(b, (pad a, n)) -> pad(sub(b, a), n)
def SubOfPadR : Pat <
  (SubPrimOp:$old $b, (PadPrimOp $a, $m)),
  (MoveNameHint $old, (PadPrimOp (SubPrimOp $b, $a), (TypeWidth32 $old))),
  [(KnownWidth $a), (KnownWidth $b)]>;

// and(x, 0) -> 0, fold can't handle all cases
def AndOfZero : Pat <
  (AndPrimOp:$old $x, (ConstantOp:$zcst $cst)),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

// and(x, -1) -> x, fold can't handle all cases
def AndOfAllOne : Pat <
  (AndPrimOp:$old $x, (ConstantOp:$ocst $cst)),
  (MoveNameHint $old, (AsUIntPrimOp (PadPrimOp $x, (TypeWidth32 $old)))),
  [(KnownWidth $x), (EqualTypes $x, $ocst), (AllOneConstantOp $ocst)]>;

/// and(x,x) -> x, fold can't handle all cases
def AndOfSelf : Pat <
  (AndPrimOp:$old $x, $x),
  (MoveNameHint $old, (AsUIntPrimOp $x)),
  [(KnownWidth $x)]>;

/// and(x, pad(y, n)) -> pad(and(tail(x), y), n), x is unsigned
def AndOfPad : Pat <
  (AndPrimOp:$old (either $x, (PadPrimOp:$pad $y, $n))),
  (MoveNameHint $old, (PadPrimOp (AndPrimOp (TailPrimOp $x, (TypeWidthAdjust32 $x, $y)), $y), $n)),
  [(KnownWidth $x), (UIntType $x), (EqualTypes $x, $pad)]>;

def AndOfAsSIntL : Pat<
  (AndPrimOp:$old (AsSIntPrimOp $x), $y),
  (MoveNameHint $old, (AndPrimOp $x, (AsUIntPrimOp $y))),
  [(KnownWidth $x), (EqualIntSize $x, $y)]>;

def AndOfAsSIntR : Pat<
  (AndPrimOp:$old $x, (AsSIntPrimOp $y)),
  (MoveNameHint $old, (AndPrimOp (AsUIntPrimOp $x), $y)),
  [(KnownWidth $x), (EqualIntSize $x, $y)]>;

// or(x, 0) -> x, fold can't handle all cases
def OrOfZero : Pat <
  (OrPrimOp:$old $x, (ConstantOp:$zcst $cst)),
  (MoveNameHint $old, (AsUIntPrimOp (PadPrimOp $x, (TypeWidth32 $old)))),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

// or(x, -1) -> -1, fold can't handle all cases
def OrOfAllOne : Pat <
  (OrPrimOp:$old $x, (ConstantOp:$ocst $cst)),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntOnesAttr($0.getType()))"> $old),
  [(KnownWidth $x), (EqualTypes $x, $ocst), (AllOneConstantOp $ocst)]>;

/// or(x,x) -> x, fold can't handle all cases
def OrOfSelf : Pat <
  (OrPrimOp:$old $x, $x),
  (MoveNameHint $old, (AsUIntPrimOp $x)),
  [(KnownWidth $x)]>;

/// or(x, pad(y, n)) -> cat(head(x), or(tail(x), y))
def OrOfPad : Pat <
  (OrPrimOp:$old $x, (PadPrimOp:$pad $y, $n)),
  (MoveNameHint $old, (CatPrimOp (HeadPrimOp $x, (TypeWidthAdjust32 $x, $y)),
                                 (OrPrimOp (TailPrimOp $x, (TypeWidthAdjust32 $x, $y)), $y))),
  [(KnownWidth $x), (UIntType $x), (EqualTypes $x, $pad)]>;

// xor(x, 0) -> x, fold can't handle all cases
def XorOfZero : Pat <
  (XorPrimOp:$old $x, (ConstantOp:$zcst $_)),
  (MoveNameHint $old, (AsUIntPrimOp (PadPrimOp $x, (TypeWidth32 $old)))),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

/// xor(x,x) -> 0, fold can't handle all cases
def XorOfSelf : Pat <
  (XorPrimOp:$old $x, $x),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(KnownWidth $x)]>;

/// xor(x, pad(y, n)) -> cat(head(x), xor(tail(x), y))
def XorOfPad : Pat <
  (XorPrimOp:$old $x, (PadPrimOp:$pad $y, $n)),
  (MoveNameHint $old, (CatPrimOp (HeadPrimOp $x, (TypeWidthAdjust32 $x, $y)),
                                 (XorPrimOp (TailPrimOp $x, (TypeWidthAdjust32 $x, $y)), $y))),
  [(KnownWidth $x), (UIntType $x), (EqualTypes $x, $pad)]>;

/// orr(asSInt(x)) -> orr(x)
def OrRasSInt : Pat <
  (OrRPrimOp:$old (AsSIntPrimOp $x)),
  (MoveNameHint $old, (OrRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// orr(asUInt(x)) -> orr(x)
def OrRasUInt : Pat <
  (OrRPrimOp:$old (AsUIntPrimOp $x)),
  (MoveNameHint $old, (OrRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// orr(cat(0,x)) -> orr(x)
def OrRCatZeroH : Pat <
  (OrRPrimOp:$old (CatPrimOp $c, $x)),
  (MoveNameHint $old, (OrRPrimOp $x)),
  [(KnownWidth $x), (ZeroConstantOp $c)]>;

/// orr(cat(x,0)) -> orr(x)
def OrRCatZeroL : Pat <
  (OrRPrimOp:$old (CatPrimOp $x, $c)),
  (MoveNameHint $old, (OrRPrimOp $x)),
  [(KnownWidth $x), (ZeroConstantOp $c)]>;

/// orr(pad(x,n)) -> orr(x)
def OrRPadU : Pat <
  (OrRPrimOp:$old (PadPrimOp:$pad $x, $n)),
  (MoveNameHint $old, (OrRPrimOp $x)),
  [(KnownWidth $x), (UIntTypes $x), (IntTypeWidthGEQ32 $pad, $x)]>;

/// xorr(asSInt(x)) -> xorr(x)
def XorRasSInt : Pat <
  (XorRPrimOp:$old (AsSIntPrimOp $x)),
  (MoveNameHint $old, (XorRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// xorr(asUInt(x)) -> xorr(x)
def XorRasUInt : Pat <
  (XorRPrimOp:$old (AsUIntPrimOp $x)),
  (MoveNameHint $old, (XorRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// xorr(cat(0,x)) -> xorr(x)
def XorRCatZeroH : Pat <
  (XorRPrimOp:$old (CatPrimOp $c, $x)),
  (MoveNameHint $old, (XorRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x), (ZeroConstantOp $c)]>;

/// xorr(cat(x,0)) -> xorr(x)
def XorRCatZeroL : Pat <
  (XorRPrimOp:$old (CatPrimOp $x, $c)),
  (MoveNameHint $old, (XorRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x), (ZeroConstantOp $c)]>;

/// xorr(pad(x,n)) -> xorr(x)
def XorRPadU : Pat <
  (XorRPrimOp:$old (PadPrimOp:$pad $x, $n)),
  (MoveNameHint $old, (XorRPrimOp $x)),
  [(KnownWidth $x), (UIntTypes $x), (IntTypeWidthGEQ32 $pad, $x)]>;

/// andr(asSInt(x)) -> andr(x)
def AndRasSInt : Pat <
  (AndRPrimOp:$old (AsSIntPrimOp $x)),
  (MoveNameHint $old, (AndRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// andr(asUInt(x)) -> andr(x)
def AndRasUInt : Pat <
  (AndRPrimOp:$old (AsUIntPrimOp $x)),
  (MoveNameHint $old, (AndRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// andr(cat(*0*, x)) -> 0
def AndRCatZeroL : Pat <
  (AndRPrimOp:$old (CatPrimOp $z, $x)),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(NotAllOneConstantOp $z)]>;

/// andr(cat(x, *0*)) -> 0
def AndRCatZeroR : Pat <
  (AndRPrimOp:$old (CatPrimOp $x, $z)),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(NotAllOneConstantOp $z)]>;

/// andr(cat(1, x)) -> andr(x)
def AndRCatOneL : Pat <
  (AndRPrimOp:$old (CatPrimOp $z, $x)),
  (MoveNameHint $old, (AndRPrimOp $x)),
  [(AllOneConstantOp $z)]>;

/// andr(cat(x, 1)) -> andr(x)
def AndRCatOneR : Pat <
  (AndRPrimOp:$old (CatPrimOp $x, $z)),
  (MoveNameHint $old, (AndRPrimOp $x)),
  [(AllOneConstantOp $z)]>;

/// andr(pad(x:U,n)) -> 0 (where pad is doing something)
def AndRPadU : Pat <
  (AndRPrimOp:$old (PadPrimOp:$pad $x, $n)),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(KnownWidth $x), (UIntTypes $x), (IntTypeWidthGT32 $pad, $x)]>;

/// andr(pad(x:S,n)) -> andr(x)
def AndRPadS : Pat <
  (AndRPrimOp:$old (PadPrimOp:$pad $x, $n)),
  (MoveNameHint $old, (AndRPrimOp $x)),
  [(KnownWidth $x), (SIntTypes $x)]>;

// bits(bits(x, ...), ...) -> bits(x, ...)
def BitsOfBits : Pat<
  (BitsPrimOp:$old (BitsPrimOp $x, I32Attr:$iHigh, I32Attr:$iLow), I32Attr:$oHigh, I32Attr:$oLow),
  (MoveNameHint $old, (BitsPrimOp $x,
    (NativeCodeCall<"$_builder.getI32IntegerAttr($0.getValue().getZExtValue() + $1.getValue().getZExtValue())"> $iLow, $oHigh),
    (NativeCodeCall<"$_builder.getI32IntegerAttr($0.getValue().getZExtValue() + $1.getValue().getZExtValue())"> $oLow, $iLow))),
  []>;

// bits(mux(c,x,y)) -> mux(c, bits(x), bits(y))
// This isn't obviously good if the mux has multiple uses, so limit it to constants.
def BitsOfMux : Pat<
  (BitsPrimOp:$old (MuxPrimOp $c, $x, $y), I32Attr:$oHigh, I32Attr:$oLow),
  (MoveNameHint $old, (MuxPrimOp $c, (BitsPrimOp $x, $oHigh, $oLow), (BitsPrimOp $y, $oHigh, $oLow))),
  [(AnyConstantOp $x), (AnyConstantOp $y)]>;

// bits(asUInt) -> bits
def BitsOfAsUInt : Pat<
  (BitsPrimOp:$old (AsUIntPrimOp $x), I32Attr:$oHigh, I32Attr:$oLow),
  (MoveNameHint $old, (BitsPrimOp $x, $oHigh, $oLow)),
  [(KnownWidth $x)]>;

// bits(asUInt) -> bits
def BitsOfAnd : Pat<
  (BitsPrimOp:$old (AndPrimOp $x, $y), I32Attr:$oHigh, I32Attr:$oLow),
  (MoveNameHint $old, (AndPrimOp (BitsPrimOp $x, $oHigh, $oLow), (BitsPrimOp $y, $oHigh, $oLow))),
  [(KnownWidth $x), (EqualTypes $x, $y)]>;

// bits(pad(x)) -> x when they cancel out
def AttrIsZero : Constraint<CPred<"$0.getValue().isZero()">>;
def BitsOfPad : Pat<
  (BitsPrimOp:$old (PadPrimOp $x, I32Attr:$n), I32Attr:$oHigh, I32Attr:$oLow),
  (AsUIntPrimOp $x),
  [(KnownWidth $x), (EqualIntSize $old, $x), (AttrIsZero $oLow)]>;

// subaccess a, cst -> subindex a, cst
// TODO: only enable if cst is inside a.  Subaccess and subindex behave differently for out-of-bounds indexes.
def SubaccessOfConstant : Pat<
  (SubaccessOp:$old $array, (ConstantOp $cst)),
  (MoveNameHint $old, (SubindexOp $array, (NativeCodeCall<"$_builder.getI32IntegerAttr(cast<IntegerAttr>($0).getAPSInt().getExtValue())"> $cst))),
  []>;

// mux(cond, 0, 1) -> ~cond
def MuxNot : Pat<
  (MuxPrimOp:$old $cond, (ConstantOp:$zcst $_), (ConstantOp:$ocst $_)),
  (MoveNameHint $old, (NotPrimOp $cond)), [
    (EqualTypes $cond, $zcst), (EqualTypes $cond, $ocst),
    (ZeroConstantOp $zcst) , (OneConstantOp $ocst)
  ]>;

// mux(cond1, t, mux(cond2, t ,f)) -> mux(or(cond1, cond2), t, f)
def MuxSameTrue : Pat <
  (MuxPrimOp:$old $cond1, $t, (MuxPrimOp $cond2, $t, $f)),
  (MoveNameHint $old, (MuxPrimOp (OrPrimOp $cond1, $cond2), $t, $f)),
  [(EqualTypes $t, $f), (KnownWidth $t)]>;

// mux(cond1, mux(cond2, t, f), f) -> mux(and(cond1, cond2), t, f)
def MuxSameFalse : Pat <
  (MuxPrimOp:$old $cond1, (MuxPrimOp $cond2, $t, $f), $f),
  (MoveNameHint $old, (MuxPrimOp (AndPrimOp $cond1, $cond2), $t, $f)),
  [(EqualTypes $t, $f), (KnownWidth $f)]>;

// mux(cond, (Concat a, b), (Concat a, c)) -> Concat(a, mux(cond, b, c))
def NarrowMuxLHS : Pat <
  (MuxPrimOp:$old $cond, (CatPrimOp:$lhs $a, $b), (CatPrimOp:$rhs $a, $c)),
  (MoveNameHint $old, (CatPrimOp $a, (MuxPrimOp $cond, $b, $c))),
  [(EqualTypes $lhs, $rhs), (KnownWidth $lhs)]>;

// mux(cond, (Concat a, b), (Concat c, b)) -> Concat(mux(cond, a, c), b)
def NarrowMuxRHS : Pat <
  (MuxPrimOp:$old $cond, (CatPrimOp:$lhs $a, $b), (CatPrimOp:$rhs $c, $b)),
  (MoveNameHint $old, (CatPrimOp (MuxPrimOp $cond, $a, $c), $b)),
  [(EqualTypes $lhs, $rhs), (KnownWidth $lhs)]>;

def CatDoubleConst : Pat <
  (CatPrimOp:$old $cst1, (CatPrimOp $cst2, $v)),
  (MoveNameHint $old, (CatPrimOp (CatPrimOp $cst1, (AsUIntPrimOp $cst2)), (AsUIntPrimOp $v))),
  [(KnownWidth $v), (AnyConstantOp $cst1), (AnyConstantOp $cst2)]>;

// regreset(clock, constant_zero, resetValue) -> reg(clock)
def RegResetWithZeroReset : Pat<
  (RegResetOp $clock, $reset, $_, $name, $nameKind, $annotations, $inner_sym, $forceable),
  (RegOp $clock, $name, $nameKind, $annotations, $inner_sym, $forceable), [(ZeroConstantOp $reset)]>;

// ref.resolve(ref.send(a)) -> a
def RefResolveOfRefSend : Pat<
  (RefResolveOp:$resolve (RefSendOp $base)),
  (replaceWithValue $base), [(EqualTypes $base, $resolve)]
>;

def UtoStoU : Pat<
  (AsSIntPrimOp (AsUIntPrimOp $x)),
  (replaceWithValue $x),
  []>;

def StoUtoS : Pat<
  (AsUIntPrimOp (AsSIntPrimOp $x)),
  (replaceWithValue $x),
  []>;

def CVTSigned : Pat<
  (CvtPrimOp $x),
  (replaceWithValue $x),
  [(SIntType $x)]>;

def CVTUnSigned : Pat<
  (CvtPrimOp:$old $x),
  (MoveNameHint $old,
    (AsSIntPrimOp
      (PadPrimOp $x,
        (NativeCodeCall<"$_builder.getI32IntegerAttr(type_cast<SIntType>($0.getType()).getBitWidthOrSentinel())"> $old)))),
  [(UIntType $x), (KnownWidth $old)]>;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD
