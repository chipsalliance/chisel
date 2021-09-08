// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>
package firrtl.backends.experimental.smt

private object SMTExprMap {
  def mapExpr(expr: SMTExpr, f: SMTExpr => SMTExpr): SMTExpr = {
    val bv = (b: BVExpr) => f(b).asInstanceOf[BVExpr]
    val ar = (a: ArrayExpr) => f(a).asInstanceOf[ArrayExpr]
    expr match {
      case b: BVExpr    => mapExpr(b, bv, ar)
      case a: ArrayExpr => mapExpr(a, bv, ar)
    }
  }

  /** maps bv/ar over subexpressions of expr and returns expr with the results replaced */
  def mapExpr(expr: BVExpr, bv: BVExpr => BVExpr, ar: ArrayExpr => ArrayExpr): BVExpr = expr match {
    // nullary
    case old: BVLiteral => old
    case old: BVSymbol  => old
    // unary
    case old @ BVExtend(e, by, signed) => val n = bv(e); if (n.eq(e)) old else BVExtend(n, by, signed)
    case old @ BVSlice(e, hi, lo)      => val n = bv(e); if (n.eq(e)) old else BVSlice(n, hi, lo)
    case old @ BVNot(e)                => val n = bv(e); if (n.eq(e)) old else BVNot(n)
    case old @ BVNegate(e)             => val n = bv(e); if (n.eq(e)) old else BVNegate(n)
    case old @ BVForall(variables, e)  => val n = bv(e); if (n.eq(e)) old else BVForall(variables, n)
    case old @ BVReduceAnd(e)          => val n = bv(e); if (n.eq(e)) old else BVReduceAnd(n)
    case old @ BVReduceOr(e)           => val n = bv(e); if (n.eq(e)) old else BVReduceOr(n)
    case old @ BVReduceXor(e) => val n = bv(e); if (n.eq(e)) old else BVReduceXor(n)
    // binary
    case old @ BVEqual(a, b) =>
      val (nA, nB) = (bv(a), bv(b)); if (nA.eq(a) && nB.eq(b)) old else BVEqual(nA, nB)
    case old @ ArrayEqual(a, b) =>
      val (nA, nB) = (ar(a), ar(b)); if (nA.eq(a) && nB.eq(b)) old else ArrayEqual(nA, nB)
    case old @ BVComparison(op, a, b, signed) =>
      val (nA, nB) = (bv(a), bv(b)); if (nA.eq(a) && nB.eq(b)) old else BVComparison(op, nA, nB, signed)
    case old @ BVOp(op, a, b) =>
      val (nA, nB) = (bv(a), bv(b)); if (nA.eq(a) && nB.eq(b)) old else BVOp(op, nA, nB)
    case old @ BVConcat(a, b) =>
      val (nA, nB) = (bv(a), bv(b)); if (nA.eq(a) && nB.eq(b)) old else BVConcat(nA, nB)
    case old @ ArrayRead(a, b) =>
      val (nA, nB) = (ar(a), bv(b)); if (nA.eq(a) && nB.eq(b)) old else ArrayRead(nA, nB)
    case old @ BVImplies(a, b) =>
      val (nA, nB) = (bv(a), bv(b)); if (nA.eq(a) && nB.eq(b)) old else BVImplies(nA, nB)
    // ternary
    case old @ BVIte(a, b, c) =>
      val (nA, nB, nC) = (bv(a), bv(b), bv(c))
      if (nA.eq(a) && nB.eq(b) && nC.eq(c)) old else BVIte(nA, nB, nC)
    // n-ary
    case old @ BVFunctionCall(name, args, width) =>
      val nArgs = args.map {
        case b: BVExpr    => bv(b)
        case a: ArrayExpr => ar(a)
        case u: UTSymbol  => u
      }
      val anyNew = nArgs.zip(args).exists { case (n, o) => !n.eq(o) }
      if (anyNew) BVFunctionCall(name, nArgs, width) else old
    case old @ BVAnd(terms) =>
      val nTerms = terms.map(bv)
      val anyNew = nTerms.zip(terms).exists { case (n, o) => !n.eq(o) }
      if (anyNew) BVAnd(nTerms) else old
    case old @ BVOr(terms) =>
      val nTerms = terms.map(bv)
      val anyNew = nTerms.zip(terms).exists { case (n, o) => !n.eq(o) }
      if (anyNew) BVOr(nTerms) else old
  }

  /** maps bv/ar over subexpressions of expr and returns expr with the results replaced */
  def mapExpr(expr: ArrayExpr, bv: BVExpr => BVExpr, ar: ArrayExpr => ArrayExpr): ArrayExpr = expr match {
    case old: ArraySymbol => old
    case old @ ArrayConstant(e, indexWidth) => val n = bv(e); if (n.eq(e)) old else ArrayConstant(n, indexWidth)
    case old @ ArrayStore(a, b, c) =>
      val (nA, nB, nC) = (ar(a), bv(b), bv(c))
      if (nA.eq(a) && nB.eq(b) && nC.eq(c)) old else ArrayStore(nA, nB, nC)
    case old @ ArrayIte(a, b, c) =>
      val (nA, nB, nC) = (bv(a), ar(b), ar(c))
      if (nA.eq(a) && nB.eq(b) && nC.eq(c)) old else ArrayIte(nA, nB, nC)
    case old @ ArrayFunctionCall(name, args, indexWidth, dataWidth) =>
      val nArgs = args.map {
        case b: BVExpr    => bv(b)
        case a: ArrayExpr => ar(a)
        case u: UTSymbol  => u
      }
      val anyNew = nArgs.zip(args).exists { case (n, o) => !n.eq(o) }
      if (anyNew) ArrayFunctionCall(name, nArgs, indexWidth, dataWidth) else old
  }
}
