// SPDX-License-Identifier: Apache-2.0
package chisel3.debug

import chisel3._
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class DebugDataTypesSpec extends AnyFunSpec with Matchers {

  import DebugTestCircuits.DataTypesCircuits._
  import DebugTestCircuits.{BindingChoice, PortBinding, RegBinding, WireBinding}
  import DebugTestUtils._

  // By-name `chirrtl` so a shared `lazy val` only forces inside `it`.
  private def emitAndCheck(label: String, gen: => RawModule)(patterns: Seq[String]): Unit =
    it(label)(checkIntrinsics(patterns.map(_ -> 1), emit(gen)))

  private def checkAt(chirrtl: => String, label: String)(patterns: Seq[String]): Unit =
    it(label)(checkIntrinsics(patterns.map(_ -> 1), chirrtl))

  def wrap(b: BindingChoice, t: String): String = b match {
    case PortBinding => s"IO[$t]"
    case WireBinding => s"Wire[$t]"
    case RegBinding  => s"Reg[$t]"
  }

  def clockResetFor(b: BindingChoice): Seq[String] =
    if (b == RegBinding) Seq(varPattern("IO[Clock]", "clock"), varPattern("IO[Reset]", "reset"))
    else Seq.empty

  def typeTests(b: BindingChoice): Unit = {
    def vp(typ: String, name: String, et: Option[String] = None) = varPattern(wrap(b, typ), name, et)
    def sp(typ: String, name: String, parent: String, et: Option[String] = None) =
      subfieldPattern(wrap(b, typ), name, parent, et)

    emitAndCheck(s"[$b] should annotate ground types", new TopCircuitGroundTypes(b))(
      Seq(vp("UInt<8>", "uint"), vp("SInt<8>", "sint"), vp("Bool", "bool"), vp("UInt<8>", "bits"))
        ++ clockResetFor(b)
        ++ (if (b != RegBinding) Seq(vp("Analog<1>", "analog")) else Nil)
    )

    emitAndCheck(s"[$b] should annotate bundles", new TopCircuitBundles(b))(
      Seq(
        vp("AnonymousBundle", "a"),
        vp("MyEmptyBundle", "bnd"),
        vp("MyBundle", "c"),
        sp("UInt<8>", "c.a", "c"),
        sp("SInt<8>", "c.b", "c"),
        sp("Bool", "c.c", "c")
      ) ++ clockResetFor(b)
    )

    emitAndCheck(s"[$b] should annotate nested bundles", new TopCircuitBundlesNested(b))(
      Seq(
        vp("MyNestedBundle", "a"),
        sp("Bool", "a.a", "a"),
        sp("MyBundle", "a.b", "a"),
        sp("UInt<8>", "a.b.a", "a"),
        sp("SInt<8>", "a.b.b", "a"),
        sp("Bool", "a.b.c", "a"),
        sp("MyBundle", "a.c", "a"),
        sp("UInt<8>", "a.c.a", "a"),
        sp("SInt<8>", "a.c.b", "a"),
        sp("Bool", "a.c.c", "a")
      ) ++ clockResetFor(b)
    )

    emitAndCheck(s"[$b] should annotate vecs", new TopCircuitVecs(b))(
      Seq(
        vp("SInt<23>[5]", "a"),
        sp("SInt<23>", "a[0]", "a"),
        vp("SInt<23>[3][5]", "bv"),
        sp("SInt<23>[3]", "bv[0]", "bv"),
        sp("SInt<23>", "bv[0][0]", "bv"),
        vp("AnonymousBundle[5]", "c"),
        sp("AnonymousBundle", "c[0]", "c"),
        sp("UInt<8>", "c[0].x", "c"),
        vp("MixedVec", "d"),
        sp("UInt<3>", "d.0", "d"),
        sp("SInt<10>", "d.1", "d")
      ) ++ clockResetFor(b)
    )

    emitAndCheck(s"[$b] should annotate bundle with vec", new TopCircuitBundleWithVec(b))(
      Seq(
        vp("AnonymousBundle", "a"),
        sp("UInt<8>[5]", "a.vec", "a"),
        sp("UInt<8>", "a.vec[0]", "a")
      ) ++ clockResetFor(b)
    )
  }

  describe("Clock and Reset annotations") {
    emitAndCheck("should annotate explicit clock and reset types", new TopCircuitClockReset)(
      Seq(
        varPattern("IO[Clock]", "clock"),
        varPattern("IO[Bool]", "syncReset"),
        varPattern("IO[Reset]", "reset"),
        varPattern("IO[AsyncReset]", "asyncReset")
      )
    )
    emitAndCheck("should annotate implicit clock and reset", new TopCircuitImplicitClockReset)(
      Seq(varPattern("IO[Clock]", "clock"), varPattern("IO[Bool]", "reset"))
    )
  }

  describe("Port (IO) annotations") { typeTests(PortBinding) }
  describe("Wire annotations") { typeTests(WireBinding) }
  describe("Reg annotations") { typeTests(RegBinding) }

  describe("ChiselEnum annotations") {
    lazy val chirrtl = emit(new TopCircuitEnumSimple)
    val DTE = Some("DebugTestEnum")
    val DTE2 = Some("DebugTestEnum2")

    checkAt(chirrtl, "should emit enumdef once per enum type")(
      Seq(enumDefPattern("DebugTestEnum"), enumDefPattern("DebugTestEnum2"))
    )
    checkAt(chirrtl, "should annotate enum port with enumTypeName")(
      Seq(varPattern("IO[DebugTestEnum]", "e", DTE), varPattern("IO[DebugTestEnum2]", "e2", DTE2))
    )
    checkAt(chirrtl, "should annotate enum subfield with enumTypeName")(
      Seq(
        subfieldPattern("IO[DebugTestEnum]", "bnd.en", "bnd", DTE),
        subfieldPattern("IO[DebugTestEnum2]", "bnd.en2", "bnd", DTE2),
        subfieldPattern("IO[UInt<8>]", "bnd.x", "bnd", None)
      )
    )
    checkAt(chirrtl, "should annotate enum in vec with enumTypeName")(
      Seq(
        varPattern("IO[DebugTestEnum[3]]", "v"),
        subfieldPattern("IO[DebugTestEnum]", "v[0]", "v", DTE)
      )
    )
  }

  describe("Subfield parent is root variable FQN") {
    // Regression guard for FG2.5b: CIRCT's CirctDebugVarConverter matches each
    // `circt_debug_subfield` leaf to its enclosing `circt_debug_var` by exact
    // name equality on the `parent` attribute. If `parent` carries the immediate
    // enclosing bundle (e.g. "io.a.b") instead of the root variable ("io"),
    // deeply nested leaves are silently filtered out and downstream attributes
    // (enumTypeName, enumFqn) are lost in the UHDI/HGLDD pipeline.
    emitAndCheck("emits parent=<root> for deeply nested leaves", new TopCircuitDeeplyNested)(
      Seq(
        varPattern("IO[MyDeeplyNestedBundle]", "io"),
        subfieldPattern("IO[AnonymousBundle]", "io.a", "io"),
        subfieldPattern("IO[AnonymousBundle]", "io.a.b", "io"),
        subfieldPattern("IO[UInt<8>]", "io.a.b.c", "io")
      )
    )
  }

  describe("Memory annotations") {
    import DebugTestCircuits.MemCircuits._

    val memVar = """intrinsic\(circt_debug_var<[^)]*name\s*=\s*"mem""""

    it("should annotate Mem as var without parent") {
      val chirrtl = emit(new TopCircuitMem(UInt(8.W)))
      countOccurrences(chirrtl, memVar) should be(1)
      countOccurrences(chirrtl, """circt_debug_subfield""") should be(0)
    }

    it("should annotate SyncReadMem as var without parent") {
      val chirrtl = emit(new TopCircuitSyncMem(UInt(8.W)))
      countOccurrences(chirrtl, memVar) should be(1)
    }
  }

  describe("MixedVec opaque-ctor handling") {
    // Per-element subfield emission already covers MixedVec's `Seq[Data]` ctor arg.
    it("does not attach a `params=` attribute to the MixedVec circt_debug_var") {
      import DebugTestCircuits.DataTypesCircuits._
      val chirrtl = emit(new TopCircuitVecs(DebugTestCircuits.PortBinding))
      val mixedVecVar = """intrinsic\(circt_debug_var<[^>]*name\s*=\s*"d"[^>]*>""".r
      val matched = mixedVecVar
        .findFirstIn(chirrtl)
        .getOrElse(
          fail("no circt_debug_var for MixedVec port `d`")
        )
      (matched should not).include("params")
    }
  }

  describe("Tmp values in when/else") {
    emitAndCheck("should annotate named vals inside when blocks", new TopCircuitWhenElse)(
      Seq(
        varPattern("OpResult[UInt<8>]", "evenSel"),
        varPattern("OpResult[UInt<8>]", "selIsOne"),
        varPattern("OpResult[UInt<8>]", "oddSel")
      )
    )
  }
}
