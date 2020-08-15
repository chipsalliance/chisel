// See LICENSE for license details.

package firrtl.ir

import firrtl.PrimOps._
import org.scalatest.flatspec.AnyFlatSpec

class StructuralHashSpec extends AnyFlatSpec {
  private def hash(n: DefModule):  HashCode = StructuralHash.sha256(n, n => n)
  private def hash(c: Circuit):    HashCode = StructuralHash.sha256Node(c)
  private def hash(e: Expression): HashCode = StructuralHash.sha256Node(e)
  private def hash(t: Type):       HashCode = StructuralHash.sha256Node(t)
  private def hash(s: Statement):  HashCode = StructuralHash.sha256Node(s)
  private val highFirrtlCompiler = new firrtl.stage.transforms.Compiler(
    targets = firrtl.stage.Forms.HighForm
  )
  private def parse(circuit: String): Circuit = {
    val rawFirrtl = firrtl.Parser.parse(circuit)
    // TODO: remove requirement that Firrtl needs to be type checked.
    //       The only reason this is needed for the structural hash right now is because we
    //       define bundles with the same list of field types to be the same, regardless of the
    //       name of these fields. Thus when the fields are accessed, we need to know their position
    //       in order to appropriately hash them.
    highFirrtlCompiler.transform(firrtl.CircuitState(rawFirrtl, Seq())).circuit
  }

  private val b0 = UIntLiteral(0, IntWidth(1))
  private val b1 = UIntLiteral(1, IntWidth(1))
  private val add = DoPrim(Add, Seq(b0, b1), Seq(), UnknownType)

  it should "generate the same hash if the objects are structurally the same" in {
    assert(hash(b0) == hash(UIntLiteral(0, IntWidth(1))))
    assert(hash(b0) != hash(UIntLiteral(1, IntWidth(1))))
    assert(hash(b0) != hash(UIntLiteral(1, IntWidth(2))))

    assert(hash(b1) == hash(UIntLiteral(1, IntWidth(1))))
    assert(hash(b1) != hash(UIntLiteral(0, IntWidth(1))))
    assert(hash(b1) != hash(UIntLiteral(1, IntWidth(2))))
  }

  it should "ignore expression types" in {
    assert(hash(add) == hash(DoPrim(Add, Seq(b0, b1), Seq(), UnknownType)))
    assert(hash(add) == hash(DoPrim(Add, Seq(b0, b1), Seq(), UIntType(UnknownWidth))))
    assert(hash(add) != hash(DoPrim(Add, Seq(b0, b0), Seq(), UnknownType)))
  }

  it should "ignore variable names" in {
    val a =
      """circuit a:
        |  module a:
        |    input x : UInt<1>
        |    output y: UInt<1>
        |    y <= x
        |""".stripMargin

    assert(hash(parse(a)) == hash(parse(a)), "the same circuit should always be equivalent")

    val b =
      """circuit a:
        |  module a:
        |    input abc : UInt<1>
        |    output haha: UInt<1>
        |    haha <= abc
        |""".stripMargin

    assert(hash(parse(a)) == hash(parse(b)), "renaming ports should not affect the hash by default")

    val c =
      """circuit a:
        |  module a:
        |    input x : UInt<1>
        |    output y: UInt<1>
        |    y <= and(x, UInt<1>(0))
        |""".stripMargin

    assert(hash(parse(a)) != hash(parse(c)), "changing an expression should affect the hash")

    val d =
      """circuit c:
        |  module c:
        |    input abc : UInt<1>
        |    output haha: UInt<1>
        |    haha <= abc
        |""".stripMargin

    assert(hash(parse(a)) != hash(parse(d)), "circuits with different names are always different")
    assert(
      hash(parse(a).modules.head) == hash(parse(d).modules.head),
      "modules with different names can be structurally different"
    )

    // for the Dedup pass we do need a way to take the port names into account
    assert(
      StructuralHash.sha256WithSignificantPortNames(parse(a).modules.head) !=
        StructuralHash.sha256WithSignificantPortNames(parse(b).modules.head),
      "renaming ports does affect the hash if we ask to"
    )
  }

  it should "not ignore port names if asked to" in {
    val e =
      """circuit a:
        |  module a:
        |    input x : UInt<1>
        |    wire y: UInt<1>
        |    y <= x
        |""".stripMargin

    val f =
      """circuit a:
        |  module a:
        |    input z : UInt<1>
        |    wire y: UInt<1>
        |    y <= z
        |""".stripMargin

    val g =
      """circuit a:
        |  module a:
        |    input x : UInt<1>
        |    wire z: UInt<1>
        |    z <= x
        |""".stripMargin

    assert(
      StructuralHash.sha256WithSignificantPortNames(parse(e).modules.head) !=
        StructuralHash.sha256WithSignificantPortNames(parse(f).modules.head),
      "renaming ports does affect the hash if we ask to"
    )
    assert(
      StructuralHash.sha256WithSignificantPortNames(parse(e).modules.head) ==
        StructuralHash.sha256WithSignificantPortNames(parse(g).modules.head),
      "renaming internal wires should never affect the hash"
    )
    assert(
      hash(parse(e).modules.head) == hash(parse(g).modules.head),
      "renaming internal wires should never affect the hash"
    )
  }

  it should "not ignore port bundle names if asked to" in {
    val e =
      """circuit a:
        |  module a:
        |    input x : {x: UInt<1>}
        |    wire y: {x: UInt<1>}
        |    y.x <= x.x
        |""".stripMargin

    val f =
      """circuit a:
        |  module a:
        |    input x : {z: UInt<1>}
        |    wire y: {x: UInt<1>}
        |    y.x <= x.z
        |""".stripMargin

    val g =
      """circuit a:
        |  module a:
        |    input x : {x: UInt<1>}
        |    wire y: {z: UInt<1>}
        |    y.z <= x.x
        |""".stripMargin

    assert(
      hash(parse(e).modules.head) == hash(parse(f).modules.head),
      "renaming port bundles does normally not affect the hash"
    )
    assert(
      StructuralHash.sha256WithSignificantPortNames(parse(e).modules.head) !=
        StructuralHash.sha256WithSignificantPortNames(parse(f).modules.head),
      "renaming port bundles does affect the hash if we ask to"
    )
    assert(
      StructuralHash.sha256WithSignificantPortNames(parse(e).modules.head) ==
        StructuralHash.sha256WithSignificantPortNames(parse(g).modules.head),
      "renaming internal wire bundles should never affect the hash"
    )
    assert(
      hash(parse(e).modules.head) == hash(parse(g).modules.head),
      "renaming internal wire bundles should never affect the hash"
    )
  }

  it should "fail on Info" in {
    // it does not make sense to hash Info nodes
    assertThrows[RuntimeException] {
      StructuralHash.sha256Node(FileInfo(StringLit("")))
    }
  }

  "Bundles with different field names" should "be structurally equivalent" in {
    def parse(str: String): BundleType = {
      val src =
        s"""circuit c:
           |  module c:
           |    input z: $str
           |""".stripMargin
      val c = firrtl.Parser.parse(src)
      val tpe = c.modules.head.ports.head.tpe
      tpe.asInstanceOf[BundleType]
    }

    val a = "{x: UInt<1>, y: UInt<1>}"
    assert(hash(parse(a)) == hash(parse(a)), "the same bundle should always be equivalent")

    val b = "{z: UInt<1>, y: UInt<1>}"
    assert(hash(parse(a)) == hash(parse(b)), "changing a field name should maintain equivalence")

    val c = "{x: UInt<2>, y: UInt<1>}"
    assert(hash(parse(a)) != hash(parse(c)), "changing a field type should not maintain equivalence")

    val d = "{x: UInt<1>, y: {y: UInt<1>}}"
    assert(hash(parse(a)) != hash(parse(d)), "changing the structure should not maintain equivalence")

    assert(hash(parse("{z: {y: {x: UInt<1>}}, a: UInt<1>}")) == hash(parse("{a: {b: {c: UInt<1>}}, z: UInt<1>}")))
  }

  "ExtModules with different names but the same defname" should "be structurally equivalent" in {
    val a =
      """circuit a:
        |  extmodule a:
        |    input x : UInt<1>
        |    defname = xyz
        |""".stripMargin

    val b =
      """circuit b:
        |  extmodule b:
        |    input y : UInt<1>
        |    defname = xyz
        |""".stripMargin

    // Q: should extmodule portnames always be significant since they map to the verilog pins?
    // A: It would be a bug for two exmodules in the same circuit to have the same defname but different
    //    port names. This should be detected by an earlier pass and thus we do not have to deal with that situation.
    assert(
      hash(parse(a).modules.head) == hash(parse(b).modules.head),
      "two ext modules with the same defname and the same type and number of ports"
    )
    assert(
      StructuralHash.sha256WithSignificantPortNames(parse(a).modules.head) !=
        StructuralHash.sha256WithSignificantPortNames(parse(b).modules.head),
      "two ext modules with significant port names"
    )
  }

  "Blocks and empty statements" should "not affect structural equivalence" in {
    val stmtA = DefNode(NoInfo, "a", UIntLiteral(1))
    val stmtB = DefNode(NoInfo, "b", UIntLiteral(1))

    val a = Block(Seq(Block(Seq(stmtA)), stmtB))
    val b = Block(Seq(stmtA, stmtB))
    assert(hash(a) == hash(b))

    val c = Block(Seq(Block(Seq(Block(Seq(stmtA, stmtB))))))
    assert(hash(a) == hash(c))

    val d = Block(Seq(stmtA))
    assert(hash(a) != hash(d))

    val e = Block(Seq(Block(Seq(stmtB)), stmtB))
    assert(hash(a) != hash(e))

    val f = Block(Seq(Block(Seq(Block(Seq(stmtA, EmptyStmt, stmtB))))))
    assert(hash(a) == hash(f))
  }

  "Conditionally" should "properly separate if and else branch" in {
    val stmtA = DefNode(NoInfo, "a", UIntLiteral(1))
    val stmtB = DefNode(NoInfo, "b", UIntLiteral(1))
    val cond = UIntLiteral(1)

    val a = Conditionally(NoInfo, cond, stmtA, stmtB)
    val b = Conditionally(NoInfo, cond, Block(Seq(stmtA)), stmtB)
    assert(hash(a) == hash(b))

    val c = Conditionally(NoInfo, cond, Block(Seq(stmtA)), Block(Seq(EmptyStmt, stmtB)))
    assert(hash(a) == hash(c))

    val d = Block(Seq(Conditionally(NoInfo, cond, stmtA, EmptyStmt), stmtB))
    assert(hash(a) != hash(d))

    val e = Conditionally(NoInfo, cond, stmtA, EmptyStmt)
    val f = Conditionally(NoInfo, cond, EmptyStmt, stmtA)
    assert(hash(e) != hash(f))
  }
}

private case object DebugHasher extends Hasher {
  override def update(b: Byte):        Unit = println(s"b(${b.toInt & 0xff})")
  override def update(i: Int):         Unit = println(s"i(${i})")
  override def update(l: Long):        Unit = println(s"l(${l})")
  override def update(s: String):      Unit = println(s"s(${s})")
  override def update(b: Array[Byte]): Unit = println(s"bytes(${b.map(x => x.toInt & 0xff).mkString(", ")})")
}
