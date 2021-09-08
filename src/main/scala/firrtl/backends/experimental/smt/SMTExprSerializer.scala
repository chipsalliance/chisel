// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

private object SMTExprSerializer {
  def serialize(expr: BVExpr): String = expr match {
    // nullary
    case lit: BVLiteral =>
      if (lit.width <= 8) {
        lit.width.toString + "'b" + lit.value.toString(2)
      } else {
        lit.width.toString + "'x" + lit.value.toString(16)
      }
    case BVSymbol(name, _) => name
    // unary
    case BVExtend(e, by, false)         => s"zext(${serialize(e)}, $by)"
    case BVExtend(e, by, true)          => s"sext(${serialize(e)}, $by)"
    case BVSlice(e, hi, lo) if hi == lo => s"${serialize(e)}[$hi]"
    case BVSlice(e, hi, lo)             => s"${serialize(e)}[$hi:$lo]"
    case BVNot(e)                       => s"not(${serialize(e)})"
    case BVNegate(e)                    => s"neg(${serialize(e)})"
    case BVForall(variable, e)          => s"forall(${variable.name} : bv<${variable.width}, ${serialize(e)})"
    case BVReduceAnd(e)                 => s"redand(${serialize(e)})"
    case BVReduceOr(e)                  => s"redor(${serialize(e)})"
    case BVReduceXor(e)                 => s"redxor(${serialize(e)})"
    // binary
    case BVEqual(a, b)                                   => s"eq(${serialize(a)}, ${serialize(b)})"
    case BVComparison(Compare.Greater, a, b, false)      => s"ugt(${serialize(a)}, ${serialize(b)})"
    case BVComparison(Compare.Greater, a, b, true)       => s"sgt(${serialize(a)}, ${serialize(b)})"
    case BVComparison(Compare.GreaterEqual, a, b, false) => s"ugeq(${serialize(a)}, ${serialize(b)})"
    case BVComparison(Compare.GreaterEqual, a, b, true)  => s"sgeq(${serialize(a)}, ${serialize(b)})"
    case BVOp(op, a, b)                                  => s"$op(${serialize(a)}, ${serialize(b)})"
    case BVConcat(a, b)                                  => s"concat(${serialize(a)}, ${serialize(b)})"
    case ArrayRead(array, index)                         => s"${serialize(array)}[${serialize(index)}]"
    case ArrayEqual(a, b)                                => s"eq(${serialize(a)}, ${serialize(b)})"
    case BVImplies(a, b)                                 => s"implies(${serialize(a)}, ${serialize(b)})"
    // ternary
    case BVIte(cond, tru, fals) => s"ite(${serialize(cond)}, ${serialize(tru)}, ${serialize(fals)})"
    // n-ary
    case BVFunctionCall(name, args, _) => name + serialize(args).mkString("(", ",", ")")
    case BVAnd(terms)                  => terms.map(serialize).mkString("and(", ", ", ")")
    case BVOr(terms)                   => terms.map(serialize).mkString("or(", ", ", ")")
  }

  def serialize(expr: ArrayExpr): String = expr match {
    case ArraySymbol(name, _, _)             => name
    case ArrayConstant(e, indexWidth)        => s"([${serialize(e)}] x ${(BigInt(1) << indexWidth)})"
    case ArrayStore(array, index, data)      => s"${serialize(array)}[${serialize(index)} := ${serialize(data)}]"
    case ArrayIte(cond, tru, fals)           => s"ite(${serialize(cond)}, ${serialize(tru)}, ${serialize(fals)})"
    case ArrayFunctionCall(name, args, _, _) => name + serialize(args).mkString("(", ",", ")")
  }

  private def serialize(args: Iterable[SMTFunctionArg]): Iterable[String] =
    args.map {
      case b: BVExpr    => serialize(b)
      case a: ArrayExpr => serialize(a)
      case u: UTSymbol  => u.name
    }
}
