// SPDX-License-Identifier: Apache-2.0
package chisel3.debug

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import chisel3._

// Method-local classes become `$anon$1`; staticClass lookup needs object scope.
object DebugCtorParamExtractorSpec {
  class Foo(val width: Int)
  class Bar(val n: Int, val label: String, val flag: Boolean)
  class Baz(x: Int)
  class Mixed(val a: Int, b: String)
  case class MyCaseClass(n: Int, name: String)
  class Inner(val x: Int)
  class Outer(val inner: Inner)
  class A(val v: Int)
  class B(val a: A)
  class C(val b: B)
  class Empty
  class WithProtected(protected val n: Int)
  class WithPrivate(private val n: Int)
  class WithData(val gen: UInt)
  class WithSecondary(val a: Int, val b: Int) {
    def this(a: Int, b: Int, c: Int, d: Int) = this(a, b)
  }
  class WithGeneric(val xs: Seq[Int])
  class WithMap(val m: Map[String, Int])
  class WithTypeParam[T](val gen: T)
  class WithDoubles(val nan: Double, val pos: Double, val neg: Double, val regular: Double)
  class WithFloats(val nan: Float, val pos: Float, val neg: Float)
  class WithBigLong(val small: Long, val big: Long)
  class SharedSibling(val left: Inner, val right: Inner)
  // Non-val ctor: reflection finds no accessor -> falls back to `toString`.
  class HugeToString(x: Int) { override def toString: String = "x" * 100000 }
  class WithHuge(val payload: HugeToString)
}

class DebugCtorParamExtractorSpec extends AnyFunSpec with Matchers {
  import DebugCtorParamExtractorSpec._

  private def s(v: String):  Option[ujson.Value] = Some(ujson.Str(v))
  private def n(v: Any):     Option[ujson.Value] = Some(ujson.Str(v.toString))
  private def b(v: Boolean): Option[ujson.Value] = Some(ujson.Bool(v))

  describe("val parameters") {
    it("extracts name, typeName and value for a single val param") {
      CtorParamExtractor.getCtorParams(new Foo(8)) shouldEqual Seq(
        ClassParam("width", "Int", n(8))
      )
    }

    it("extracts multiple val params of different types") {
      CtorParamExtractor.getCtorParams(new Bar(4, "hello", true)) shouldEqual Seq(
        ClassParam("n", "Int", n(4)),
        ClassParam("label", "String", s("hello")),
        ClassParam("flag", "Boolean", b(true))
      )
    }
  }

  describe("non-val parameters") {
    it("returns None for value when param has no val") {
      CtorParamExtractor.getCtorParams(new Baz(42)) shouldEqual Seq(
        ClassParam("x", "Int", None)
      )
    }

    it("mixes val and non-val in same constructor") {
      CtorParamExtractor.getCtorParams(new Mixed(1, "ignored")) shouldEqual Seq(
        ClassParam("a", "Int", n(1)),
        ClassParam("b", "String", None)
      )
    }
  }

  describe("case class") {
    it("extracts all fields with values") {
      CtorParamExtractor.getCtorParams(MyCaseClass(7, "world")) shouldEqual Seq(
        ClassParam("n", "Int", n(7)),
        ClassParam("name", "String", s("world"))
      )
    }
  }

  describe("nested class parameters") {
    it("recursively serializes one level of nesting") {
      CtorParamExtractor.getCtorParams(new Outer(new Inner(3))) shouldEqual Seq(
        ClassParam("inner", "Inner", s("Inner(x: 3)"))
      )
    }

    it("handles two levels of nesting") {
      val params = CtorParamExtractor.getCtorParams(new C(new B(new A(5))))
      params.head.value shouldEqual s("B(a: A(v: 5))")
    }
  }

  describe("empty constructor") {
    it("returns empty Seq") {
      CtorParamExtractor.getCtorParams(new Empty) shouldBe empty
    }
  }

  describe("protected and private val parameters") {
    it("extracts protected val") {
      CtorParamExtractor.getCtorParams(new WithProtected(9)) shouldEqual Seq(
        ClassParam("n", "Int", n(9))
      )
    }

    it("extracts private val") {
      CtorParamExtractor.getCtorParams(new WithPrivate(9)) shouldEqual Seq(
        ClassParam("n", "Int", n(9))
      )
    }
  }

  describe("Data parameters") {
    it("does not crash and returns non-empty value for unbound Data param") {
      val params = CtorParamExtractor.getCtorParams(new WithData(UInt(8.W)))
      (params should have).length(1)
      params.head.name shouldEqual "gen"
      params.head.typeName shouldEqual "UInt"
      params.head.value should not be empty
    }
  }

  describe("type names") {
    it("strips generic arguments from a Seq type") {
      val params = CtorParamExtractor.getCtorParams(new WithGeneric(Seq(1, 2, 3)))
      params.head.typeName shouldEqual "Seq"
    }

    it("strips generic arguments from a Map type") {
      val params = CtorParamExtractor.getCtorParams(new WithMap(Map("a" -> 1)))
      params.head.typeName shouldEqual "Map"
    }

    it("returns the type-parameter name for an unbound type ref") {
      val params = CtorParamExtractor.getCtorParams(new WithTypeParam[Int](42))
      params.head.typeName shouldEqual "T"
    }
  }

  describe("numeric values") {
    // Numbers serialize as strings: JSON has no NaN/Infinity, Double loses
    // precision on Long > 2^53.
    it("emits Double NaN/Infinity and finite values uniformly as strings") {
      val params = CtorParamExtractor.getCtorParams(
        new WithDoubles(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity, 1.5)
      )
      val byName = params.map(p => p.name -> p.value).toMap
      byName("nan") shouldEqual Some(ujson.Str("NaN"))
      byName("pos") shouldEqual Some(ujson.Str("Infinity"))
      byName("neg") shouldEqual Some(ujson.Str("-Infinity"))
      byName("regular") shouldEqual Some(ujson.Str("1.5"))
    }

    it("emits Float NaN/Infinity as strings") {
      val params = CtorParamExtractor.getCtorParams(
        new WithFloats(Float.NaN, Float.PositiveInfinity, Float.NegativeInfinity)
      )
      val byName = params.map(p => p.name -> p.value).toMap
      byName("nan") shouldEqual Some(ujson.Str("NaN"))
      byName("pos") shouldEqual Some(ujson.Str("Infinity"))
      byName("neg") shouldEqual Some(ujson.Str("-Infinity"))
    }

    it("emits Long values as strings preserving exact representation") {
      val big = (1L << 53) + 1L
      val params = CtorParamExtractor.getCtorParams(new WithBigLong(42L, big))
      val byName = params.map(p => p.name -> p.value).toMap
      byName("small") shouldEqual Some(ujson.Str("42"))
      byName("big") shouldEqual Some(ujson.Str(big.toString))
    }
  }

  describe("shared sibling object (not a cycle)") {
    // `visited` is path-scoped: a shared sibling must recurse on both visits.
    it("recurses into both occurrences of a shared object") {
      val cfg = new Inner(7)
      val params = CtorParamExtractor.getCtorParams(new SharedSibling(cfg, cfg))
      params.map(p => p.name -> p.value) shouldEqual Seq(
        "left" -> s("Inner(x: 7)"),
        "right" -> s("Inner(x: 7)")
      )
    }
  }

  describe("oversized toString") {
    it("truncates pathologically long toString output") {
      val params = CtorParamExtractor.getCtorParams(new WithHuge(new HugeToString(1)))
      val rendered = params.head.value.collect { case ujson.Str(v) => v }.getOrElse("")
      rendered.length should be <= CtorParamExtractor.MaxRenderedLen + CtorParamExtractor.TruncatedSuffix.length
      rendered should endWith(CtorParamExtractor.TruncatedSuffix)
    }
  }

  describe("primary vs. secondary constructors") {
    // Scala 2 yields ("a","b"); Scala 3 returns empty. Neither must leak secondary params.
    it("never reports parameter names that don't correspond to instance accessors") {
      val params = CtorParamExtractor.getCtorParams(new WithSecondary(1, 2))
      val names = params.map(_.name)
      names should (equal(Seq("a", "b")).or(be(empty)))
      names should not contain "c"
      names should not contain "d"
    }
  }
}
