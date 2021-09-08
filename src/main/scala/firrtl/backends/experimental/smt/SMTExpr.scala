// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>
// Inspired by the uclid5 SMT library (https://github.com/uclid-org/uclid).
// And the btor2 documentation (BTOR2 , BtorMC and Boolector 3.0 by Niemetz et.al.)

package firrtl.backends.experimental.smt

/** base trait for all SMT expressions */
private sealed trait SMTExpr extends SMTFunctionArg {
  def tpe:      SMTType
  def children: List[SMTExpr]
}
private sealed trait SMTSymbol extends SMTExpr with SMTNullaryExpr {
  def name: String

  /** keeps the type of the symbol while changing the name */
  def rename(newName: String): SMTSymbol
}
private object SMTSymbol {

  /** makes a SMTSymbol of the same type as the expression */
  def fromExpr(name: String, e: SMTExpr): SMTSymbol = e match {
    case b: BVExpr    => BVSymbol(name, b.width)
    case a: ArrayExpr => ArraySymbol(name, a.indexWidth, a.dataWidth)
  }
}
private sealed trait SMTNullaryExpr extends SMTExpr {
  override def children: List[SMTExpr] = List()
}

/** a SMT bit vector expression: https://smtlib.cs.uiowa.edu/theories-FixedSizeBitVectors.shtml */
private sealed trait BVExpr extends SMTExpr {
  def width: Int
  def tpe:               BVType = BVType(width)
  override def toString: String = SMTExprSerializer.serialize(this)
}
private case class BVLiteral(value: BigInt, width: Int) extends BVExpr with SMTNullaryExpr {
  private def minWidth = value.bitLength + (if (value <= 0) 1 else 0)
  assert(value >= 0, "Negative values are not supported! Please normalize by calculating 2s complement.")
  assert(width > 0, "Zero or negative width literals are not allowed!")
  assert(width >= minWidth, "Value (" + value.toString + ") too big for BitVector of width " + width + " bits.")
}
private object BVLiteral {
  def apply(nums: String): BVLiteral = nums.head match {
    case 'b' => BVLiteral(BigInt(nums.drop(1), 2), nums.length - 1)
  }
}
private case class BVSymbol(name: String, width: Int) extends BVExpr with SMTSymbol {
  assert(!name.contains("|"), s"Invalid id $name contains escape character `|`")
  assert(width > 0, "Zero width bit vectors are not supported!")
  override def rename(newName: String) = BVSymbol(newName, width)
}

private sealed trait BVUnaryExpr extends BVExpr {
  def e: BVExpr

  /** same function, different child, e.g.: not(x) -- reapply(Y) --> not(Y) */
  def reapply(expr: BVExpr): BVUnaryExpr
  override def children: List[BVExpr] = List(e)
}
private case class BVExtend(e: BVExpr, by: Int, signed: Boolean) extends BVUnaryExpr {
  assert(by >= 0, "Extension must be non-negative!")
  override val width: Int = e.width + by
  override def reapply(expr: BVExpr) = BVExtend(expr, by, signed)
}
// also known as bit extract operation
private case class BVSlice(e: BVExpr, hi: Int, lo: Int) extends BVUnaryExpr {
  assert(lo >= 0, s"lo (lsb) must be non-negative!")
  assert(hi >= lo, s"hi (msb) must not be smaller than lo (lsb): msb: $hi lsb: $lo")
  assert(e.width > hi, s"Out off bounds hi (msb) access: width: ${e.width} msb: $hi")
  override def width: Int = hi - lo + 1
  override def reapply(expr: BVExpr) = BVSlice(expr, hi, lo)
}
private case class BVNot(e: BVExpr) extends BVUnaryExpr {
  override val width: Int = e.width
  override def reapply(expr: BVExpr) = new BVNot(expr)
}
private case class BVNegate(e: BVExpr) extends BVUnaryExpr {
  override val width: Int = e.width
  override def reapply(expr: BVExpr) = BVNegate(expr)
}

private case class BVReduceOr(e: BVExpr) extends BVUnaryExpr {
  override def width: Int = 1
  override def reapply(expr: BVExpr) = BVReduceOr(expr)
}
private case class BVReduceAnd(e: BVExpr) extends BVUnaryExpr {
  override def width: Int = 1
  override def reapply(expr: BVExpr) = BVReduceAnd(expr)
}
private case class BVReduceXor(e: BVExpr) extends BVUnaryExpr {
  override def width: Int = 1
  override def reapply(expr: BVExpr) = BVReduceXor(expr)
}

private sealed trait BVBinaryExpr extends BVExpr {
  def a: BVExpr
  def b: BVExpr
  override def children: List[BVExpr] = List(a, b)

  /** same function, different child, e.g.: add(a,b) -- reapply(a,c) --> add(a,c) */
  def reapply(nA: BVExpr, nB: BVExpr): BVBinaryExpr
}
private case class BVEqual(a: BVExpr, b: BVExpr) extends BVBinaryExpr {
  assert(a.width == b.width, s"Both argument need to be the same width!")
  override def width: Int = 1
  override def reapply(nA: BVExpr, nB: BVExpr) = BVEqual(nA, nB)
}
// added as a separate node because it is used a lot in model checking and benefits from pretty printing
private class BVImplies(val a: BVExpr, val b: BVExpr) extends BVBinaryExpr {
  assert(a.width == 1, s"The antecedent needs to be a boolean expression!")
  assert(b.width == 1, s"The consequent needs to be a boolean expression!")
  override def width: Int = 1
  override def reapply(nA: BVExpr, nB: BVExpr) = new BVImplies(nA, nB)
}
private object BVImplies {
  def apply(a: BVExpr, b: BVExpr): BVExpr = {
    assert(a.width == b.width, s"Both argument need to be the same width!")
    (a, b) match {
      case (True(), b)  => b // (!1 || b) = b
      case (False(), _) => True() // (!0 || _) = (1 || _) = 1
      case (_, True())  => True() // (!a || 1) = 1
      case (a, False()) => BVNot(a) // (!a || 0) = !a
      case (a, b)       => new BVImplies(a, b)
    }
  }
  def unapply(i: BVImplies): Some[(BVExpr, BVExpr)] = Some((i.a, i.b))
}

private object Compare extends Enumeration {
  val Greater, GreaterEqual = Value
}
private case class BVComparison(op: Compare.Value, a: BVExpr, b: BVExpr, signed: Boolean) extends BVBinaryExpr {
  assert(a.width == b.width, s"Both argument need to be the same width!")
  override def width: Int = 1
  override def reapply(nA: BVExpr, nB: BVExpr) = BVComparison(op, nA, nB, signed)
}

private object Op extends Enumeration {
  val Xor = Value("xor")
  val ShiftLeft = Value("logical_shift_left")
  val ArithmeticShiftRight = Value("arithmetic_shift_right")
  val ShiftRight = Value("logical_shift_right")
  val Add = Value("add")
  val Mul = Value("mul")
  val SignedDiv = Value("sdiv")
  val UnsignedDiv = Value("udiv")
  val SignedMod = Value("smod")
  val SignedRem = Value("srem")
  val UnsignedRem = Value("urem")
  val Sub = Value("sub")
}
private case class BVOp(op: Op.Value, a: BVExpr, b: BVExpr) extends BVBinaryExpr {
  assert(a.width == b.width, s"Both argument need to be the same width!")
  override val width: Int = a.width
  override def reapply(nA: BVExpr, nB: BVExpr) = BVOp(op, nA, nB)
}
private case class BVConcat(a: BVExpr, b: BVExpr) extends BVBinaryExpr {
  override val width: Int = a.width + b.width
  override def reapply(nA: BVExpr, nB: BVExpr) = BVConcat(nA, nB)
}
private case class ArrayRead(array: ArrayExpr, index: BVExpr) extends BVExpr {
  assert(array.indexWidth == index.width, "Index with does not match expected array index width!")
  override val width:    Int = array.dataWidth
  override def children: List[SMTExpr] = List(array, index)
}
private case class BVIte(cond: BVExpr, tru: BVExpr, fals: BVExpr) extends BVExpr {
  assert(cond.width == 1, s"Condition needs to be a 1-bit value not ${cond.width}-bit!")
  assert(tru.width == fals.width, s"Both branches need to be of the same width! ${tru.width} vs ${fals.width}")
  override val width:    Int = tru.width
  override def children: List[BVExpr] = List(cond, tru, fals)
}

private case class BVAnd(terms: List[BVExpr]) extends BVExpr {
  require(terms.size > 1)
  override val width: Int = terms.head.width
  require(terms.forall(_.width == width))
  override def children: List[BVExpr] = terms
}

private case class BVOr(terms: List[BVExpr]) extends BVExpr {
  require(terms.size > 1)
  override val width: Int = terms.head.width
  require(terms.forall(_.width == width))
  override def children: List[BVExpr] = terms
}

private sealed trait ArrayExpr extends SMTExpr {
  val indexWidth: Int
  val dataWidth:  Int
  def tpe:               ArrayType = ArrayType(indexWidth = indexWidth, dataWidth = dataWidth)
  override def toString: String = SMTExprSerializer.serialize(this)
}
private case class ArraySymbol(name: String, indexWidth: Int, dataWidth: Int) extends ArrayExpr with SMTSymbol {
  assert(!name.contains("|"), s"Invalid id $name contains escape character `|`")
  assert(!name.contains("\\"), s"Invalid id $name contains `\\`")
  override def rename(newName: String) = ArraySymbol(newName, indexWidth, dataWidth)
}
private case class ArrayConstant(e: BVExpr, indexWidth: Int) extends ArrayExpr {
  override val dataWidth: Int = e.width
  override def children:  List[SMTExpr] = List(e)
}
private case class ArrayEqual(a: ArrayExpr, b: ArrayExpr) extends BVExpr {
  assert(a.indexWidth == b.indexWidth, s"Both argument need to be the same index width!")
  assert(a.dataWidth == b.dataWidth, s"Both argument need to be the same data width!")
  override def width:    Int = 1
  override def children: List[SMTExpr] = List(a, b)
}
private case class ArrayStore(array: ArrayExpr, index: BVExpr, data: BVExpr) extends ArrayExpr {
  assert(array.indexWidth == index.width, "Index with does not match expected array index width!")
  assert(array.dataWidth == data.width, "Data with does not match expected array data width!")
  override val dataWidth:  Int = array.dataWidth
  override val indexWidth: Int = array.indexWidth
  override def children:   List[SMTExpr] = List(array, index, data)
}
private case class ArrayIte(cond: BVExpr, tru: ArrayExpr, fals: ArrayExpr) extends ArrayExpr {
  assert(cond.width == 1, s"Condition needs to be a 1-bit value not ${cond.width}-bit!")
  assert(
    tru.indexWidth == fals.indexWidth,
    s"Both branches need to be of the same type! ${tru.indexWidth} vs ${fals.indexWidth}"
  )
  assert(
    tru.dataWidth == fals.dataWidth,
    s"Both branches need to be of the same type! ${tru.dataWidth} vs ${fals.dataWidth}"
  )
  override val dataWidth:  Int = tru.dataWidth
  override val indexWidth: Int = tru.indexWidth
  override def children:   List[SMTExpr] = List(cond, tru, fals)
}

private case class BVForall(variable: BVSymbol, e: BVExpr) extends BVUnaryExpr {
  assert(e.width == 1, "Can only quantify over boolean expressions!")
  override def width = 1
  override def reapply(expr: BVExpr) = BVForall(variable, expr)
}

/** apply arguments to a function which returns a result of bit vector type */
private case class BVFunctionCall(name: String, args: List[SMTFunctionArg], width: Int) extends BVExpr {
  override def children = args.map(_.asInstanceOf[SMTExpr])
}

/** apply arguments to a function which returns a result of array type */
private case class ArrayFunctionCall(name: String, args: List[SMTFunctionArg], indexWidth: Int, dataWidth: Int)
    extends ArrayExpr {
  override def children = args.map(_.asInstanceOf[SMTExpr])
}
private sealed trait SMTFunctionArg
// we allow symbols with uninterpreted type to be function arguments
private case class UTSymbol(name: String, tpe: String) extends SMTFunctionArg

private object BVAnd {
  def apply(a: BVExpr, b: BVExpr): BVExpr = {
    assert(a.width == b.width, s"Both argument need to be the same width!")
    (a, b) match {
      case (True(), b)  => b
      case (a, True())  => a
      case (False(), _) => False()
      case (_, False()) => False()
      case (a, b)       => new BVAnd(List(a, b))
    }
  }
  def apply(exprs: List[BVExpr]): BVExpr = {
    assert(exprs.nonEmpty, "Don't know what to do with an empty list!")
    val nonTriviallyTrue = exprs.filterNot(_ == True())
    nonTriviallyTrue.distinct match {
      case Seq()    => True()
      case Seq(one) => one
      case terms    => new BVAnd(terms)
    }
  }
}
private object BVOr {
  def apply(a: BVExpr, b: BVExpr): BVExpr = {
    assert(a.width == b.width, s"Both argument need to be the same width!")
    (a, b) match {
      case (True(), _)  => True()
      case (_, True())  => True()
      case (False(), b) => b
      case (a, False()) => a
      case (a, b)       => new BVOr(List(a, b))
    }
  }
  def apply(exprs: List[BVExpr]): BVExpr = {
    assert(exprs.nonEmpty, "Don't know what to do with an empty list!")
    val nonTriviallyFalse = exprs.filterNot(_ == False())
    nonTriviallyFalse.distinct match {
      case Seq()    => False()
      case Seq(one) => one
      case terms    => new BVOr(terms)
    }
  }
}

private object BVNot {
  def apply(e: BVExpr): BVExpr = e match {
    case True()       => False()
    case False()      => True()
    case BVNot(inner) => inner
    case other        => new BVNot(other)
  }
}

private object SMTEqual {
  def apply(a: SMTExpr, b: SMTExpr): BVExpr = (a, b) match {
    case (ab: BVExpr, bb: BVExpr) => BVEqual(ab, bb)
    case (aa: ArrayExpr, ba: ArrayExpr) => ArrayEqual(aa, ba)
    case _ => throw new RuntimeException(s"Cannot compare $a and $b")
  }
}

private object SMTIte {
  def apply(cond: BVExpr, tru: SMTExpr, fals: SMTExpr): SMTExpr = (tru, fals) match {
    case (ab: BVExpr, bb: BVExpr) => BVIte(cond, ab, bb)
    case (aa: ArrayExpr, ba: ArrayExpr) => ArrayIte(cond, aa, ba)
    case _ => throw new RuntimeException(s"Cannot mux $tru and $fals")
  }
}

private object SMTExpr {
  def serializeType(e: SMTExpr): String = e match {
    case b: BVExpr    => s"bv<${b.width}>"
    case a: ArrayExpr => s"bv<${a.indexWidth}> -> bv<${a.dataWidth}>"
  }
}

// unapply for matching BVLiteral(1, 1)
private object True {
  private val _True = BVLiteral(1, 1)
  def apply(): BVLiteral = _True
  def unapply(l: BVLiteral): Boolean = l.value == 1 && l.width == 1
}

// unapply for matching BVLiteral(0, 1)
private object False {
  private val _False = BVLiteral(0, 1)
  def apply(): BVLiteral = _False
  def unapply(l: BVLiteral): Boolean = l.value == 0 && l.width == 1
}

private sealed trait SMTType
private case class BVType(width: Int) extends SMTType
private case class ArrayType(indexWidth: Int, dataWidth: Int) extends SMTType
