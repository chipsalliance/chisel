package firrtlTests.constraint

import firrtl.constraint._
import firrtl.ir.Closed
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class InequalitySpec extends AnyFlatSpec with Matchers {

  behavior.of("Constraints")

  "IsConstraints" should "reduce properly" in {
    IsMin(Closed(0), Closed(1)) should be(Closed(0))
    IsMin(Closed(-1), Closed(1)) should be(Closed(-1))
    IsMax(Closed(-1), Closed(1)) should be(Closed(1))
    IsNeg(IsMul(Closed(-1), Closed(-2))) should be(Closed(-2))
    val x = IsMin(IsMul(Closed(1), VarCon("a")), Closed(2))
    x.children.toSet should be(IsMin(Closed(2), IsMul(Closed(1), VarCon("a"))).children.toSet)
  }

  "IsAdd" should "reduce properly" in {
    // All constants
    IsAdd(Closed(-1), Closed(1)) should be(Closed(0))

    // Pull Out IsMax
    IsAdd(Closed(1), IsMax(Closed(1), VarCon("a"))) should be(IsMax(Closed(2), IsAdd(VarCon("a"), Closed(1))))
    IsAdd(Closed(1), IsMax(Seq(Closed(1), VarCon("a"), VarCon("b")))) should be(
      IsMax(Seq(Closed(2), IsAdd(VarCon("a"), Closed(1)), IsAdd(VarCon("b"), Closed(1))))
    )

    // Pull Out IsMin
    IsAdd(Closed(1), IsMin(Closed(1), VarCon("a"))) should be(IsMin(Closed(2), IsAdd(VarCon("a"), Closed(1))))
    IsAdd(Closed(1), IsMin(Seq(Closed(1), VarCon("a"), VarCon("b")))) should be(
      IsMin(Seq(Closed(2), IsAdd(VarCon("a"), Closed(1)), IsAdd(VarCon("b"), Closed(1))))
    )

    // Add Zero
    IsAdd(Closed(0), VarCon("a")) should be(VarCon("a"))

    // One argument
    IsAdd(Seq(VarCon("a"))) should be(VarCon("a"))
  }

  "IsMax" should "reduce properly" in {
    // All constants
    IsMax(Closed(-1), Closed(1)) should be(Closed(1))

    // Flatten nested IsMax
    IsMax(Closed(1), IsMax(Closed(1), VarCon("a"))) should be(IsMax(Closed(1), VarCon("a")))
    IsMax(Closed(1), IsMax(Seq(Closed(1), VarCon("a"), VarCon("b")))) should be(
      IsMax(Seq(Closed(1), VarCon("a"), VarCon("b")))
    )

    // Eliminate IsMins if possible
    IsMax(Closed(2), IsMin(Closed(1), VarCon("a"))) should be(Closed(2))
    IsMax(
      Seq(
        Closed(2),
        IsMin(Closed(1), VarCon("a")),
        IsMin(Closed(3), VarCon("b"))
      )
    ) should be(
      IsMax(
        Seq(
          Closed(2),
          IsMin(Closed(3), VarCon("b"))
        )
      )
    )

    // One argument
    IsMax(Seq(VarCon("a"))) should be(VarCon("a"))
    IsMax(Seq(Closed(0))) should be(Closed(0))
    IsMax(Seq(IsMin(VarCon("a"), Closed(0)))) should be(IsMin(VarCon("a"), Closed(0)))
  }

  "IsMin" should "reduce properly" in {
    // All constants
    IsMin(Closed(-1), Closed(1)) should be(Closed(-1))

    // Flatten nested IsMin
    IsMin(Closed(1), IsMin(Closed(1), VarCon("a"))) should be(IsMin(Closed(1), VarCon("a")))
    IsMin(Closed(1), IsMin(Seq(Closed(1), VarCon("a"), VarCon("b")))) should be(
      IsMin(Seq(Closed(1), VarCon("a"), VarCon("b")))
    )

    // Eliminate IsMaxs if possible
    IsMin(Closed(1), IsMax(Closed(2), VarCon("a"))) should be(Closed(1))
    IsMin(
      Seq(
        Closed(2),
        IsMax(Closed(1), VarCon("a")),
        IsMax(Closed(3), VarCon("b"))
      )
    ) should be(
      IsMin(
        Seq(
          Closed(2),
          IsMax(Closed(1), VarCon("a"))
        )
      )
    )

    // One argument
    IsMin(Seq(VarCon("a"))) should be(VarCon("a"))
    IsMin(Seq(Closed(0))) should be(Closed(0))
    IsMin(Seq(IsMax(VarCon("a"), Closed(0)))) should be(IsMax(VarCon("a"), Closed(0)))
  }

  "IsMul" should "reduce properly" in {
    // All constants
    IsMul(Closed(2), Closed(3)) should be(Closed(6))

    // Pull out max, if positive stays max
    IsMul(Closed(2), IsMax(Closed(3), VarCon("a"))) should be(
      IsMax(Closed(6), IsMul(Closed(2), VarCon("a")))
    )

    // Pull out max, if negative is min
    IsMul(Closed(-2), IsMax(Closed(3), VarCon("a"))) should be(
      IsMin(Closed(-6), IsMul(Closed(-2), VarCon("a")))
    )

    // Pull out min, if positive stays min
    IsMul(Closed(2), IsMin(Closed(3), VarCon("a"))) should be(
      IsMin(Closed(6), IsMul(Closed(2), VarCon("a")))
    )

    // Pull out min, if negative is max
    IsMul(Closed(-2), IsMin(Closed(3), VarCon("a"))) should be(
      IsMax(Closed(-6), IsMul(Closed(-2), VarCon("a")))
    )

    // Times zero
    IsMul(Closed(0), VarCon("x")) should be(Closed(0))

    // Times 1
    IsMul(Closed(1), VarCon("x")) should be(VarCon("x"))

    // One argument
    IsMul(Seq(Closed(0))) should be(Closed(0))
    IsMul(Seq(VarCon("a"))) should be(VarCon("a"))

    // No optimizations
    val isMax = IsMax(VarCon("x"), VarCon("y"))
    val isMin = IsMin(VarCon("x"), VarCon("y"))
    val a = VarCon("a")
    IsMul(a, isMax).children should be(Vector(a, isMax)) //non-known multiply
    IsMul(a, isMin).children should be(Vector(a, isMin)) //non-known multiply
    IsMul(Seq(Closed(2), isMin, isMin)).children should be(Vector(Closed(2), isMin, isMin)) //>1 min
    IsMul(Seq(Closed(2), isMax, isMax)).children should be(Vector(Closed(2), isMax, isMax)) //>1 max
    IsMul(Seq(Closed(2), isMin, isMax)).children should be(Vector(Closed(2), isMin, isMax)) //mixed min/max
  }

  "IsNeg" should "reduce properly" in {
    // All constants
    IsNeg(Closed(1)) should be(Closed(-1))
    // Pull out max
    IsNeg(IsMax(Closed(1), VarCon("a"))) should be(IsMin(Closed(-1), IsNeg(VarCon("a"))))
    // Pull out min
    IsNeg(IsMin(Closed(1), VarCon("a"))) should be(IsMax(Closed(-1), IsNeg(VarCon("a"))))
    // Pull out add
    IsNeg(IsAdd(Closed(1), VarCon("a"))) should be(IsAdd(Closed(-1), IsNeg(VarCon("a"))))
    // Pull out mul
    IsNeg(IsMul(Closed(2), VarCon("a"))) should be(IsMul(Closed(-2), VarCon("a")))
    // No optimizations
    // (pow), (floor?)
    IsNeg(IsPow(VarCon("x"))).children should be(Vector(IsPow(VarCon("x"))))
    IsNeg(IsFloor(VarCon("x"))).children should be(Vector(IsFloor(VarCon("x"))))
  }

  "IsPow" should "reduce properly" in {
    // All constants
    IsPow(Closed(1)) should be(Closed(2))
    // Pull out max
    IsPow(IsMax(Closed(1), VarCon("a"))) should be(IsMax(Closed(2), IsPow(VarCon("a"))))
    // Pull out min
    IsPow(IsMin(Closed(1), VarCon("a"))) should be(IsMin(Closed(2), IsPow(VarCon("a"))))
    // Pull out add
    IsPow(IsAdd(Closed(1), VarCon("a"))) should be(IsMul(Closed(2), IsPow(VarCon("a"))))
    // No optimizations
    // (mul), (pow), (floor?)
    IsPow(IsMul(Closed(2), VarCon("x"))).children should be(Vector(IsMul(Closed(2), VarCon("x"))))
    IsPow(IsPow(VarCon("x"))).children should be(Vector(IsPow(VarCon("x"))))
    IsPow(IsFloor(VarCon("x"))).children should be(Vector(IsFloor(VarCon("x"))))
  }

  "IsFloor" should "reduce properly" in {
    // All constants
    IsFloor(Closed(1.9)) should be(Closed(1))
    IsFloor(Closed(-1.9)) should be(Closed(-2))
    // Pull out max
    IsFloor(IsMax(Closed(1.9), VarCon("a"))) should be(IsMax(Closed(1), IsFloor(VarCon("a"))))
    // Pull out min
    IsFloor(IsMin(Closed(1.9), VarCon("a"))) should be(IsMin(Closed(1), IsFloor(VarCon("a"))))
    // Cancel with another floor
    IsFloor(IsFloor(VarCon("a"))) should be(IsFloor(VarCon("a")))
    // No optimizations
    // (add), (mul), (pow)
    IsFloor(IsMul(Closed(2), VarCon("x"))).children should be(Vector(IsMul(Closed(2), VarCon("x"))))
    IsFloor(IsPow(VarCon("x"))).children should be(Vector(IsPow(VarCon("x"))))
    IsFloor(IsAdd(Closed(1), VarCon("x"))).children should be(Vector(IsAdd(Closed(1), VarCon("x"))))
  }

}
