// SPDX-License-Identifier: Apache-2.0

package firrtl.analysis

import firrtl.analyses._
import firrtl.ir
import firrtl.options.Dependency
import org.scalatest.flatspec.AnyFlatSpec

class SymbolTableSpec extends AnyFlatSpec {
  behavior.of("SymbolTable")

  private val src =
    """circuit m:
      |  module child:
      |    input x : UInt<2>
      |    skip
      |  module m:
      |    input clk : Clock
      |    input x : UInt<1>
      |    output y : UInt<3>
      |    wire z : SInt<1>
      |    node a = cat(asUInt(z), x)
      |    inst i of child
      |    reg r: SInt<4>, clk
      |    mem m:
      |      data-type => UInt<8>
      |      depth => 31
      |      reader => r
      |      read-latency => 1
      |      write-latency => 1
      |      read-under-write => undefined
      |""".stripMargin

  it should "find all declarations in module m before InferTypes" in {
    val c = firrtl.Parser.parse(src)
    val m = c.modules.find(_.name == "m").get

    val syms = SymbolTable.scanModule(m, new LocalSymbolTable with WithMap)
    assert(syms.size == 8)
    assert(syms("clk").tpe == ir.ClockType && syms("clk").kind == firrtl.PortKind)
    assert(syms("x").tpe == ir.UIntType(ir.IntWidth(1)) && syms("x").kind == firrtl.PortKind)
    assert(syms("y").tpe == ir.UIntType(ir.IntWidth(3)) && syms("y").kind == firrtl.PortKind)
    assert(syms("z").tpe == ir.SIntType(ir.IntWidth(1)) && syms("z").kind == firrtl.WireKind)
    // The expression type which determines the node type is only known after InferTypes.
    assert(syms("a").tpe == ir.UnknownType && syms("a").kind == firrtl.NodeKind)
    // The type of the instance is unknown because we scanned the module before InferTypes and the table
    // uses only local information.
    assert(syms("i").tpe == ir.UnknownType && syms("i").kind == firrtl.InstanceKind)
    assert(syms("r").tpe == ir.SIntType(ir.IntWidth(4)) && syms("r").kind == firrtl.RegKind)
    val mType = firrtl.passes.MemPortUtils.memType(
      // only dataType, depth and reader, writer, readwriter properties affect the data type
      ir.DefMemory(
        ir.NoInfo,
        "???",
        ir.UIntType(ir.IntWidth(8)),
        32,
        10,
        10,
        Seq("r"),
        Seq(),
        Seq(),
        ir.ReadUnderWrite.New
      )
    )
    assert(syms("m").tpe == mType && syms("m").kind == firrtl.MemKind)
  }

  it should "find all declarations in module m after InferTypes" in {
    val c = firrtl.Parser.parse(src)
    val inferTypesCompiler = new firrtl.stage.TransformManager(Seq(Dependency(firrtl.passes.InferTypes)))
    val inferredC = inferTypesCompiler.execute(firrtl.CircuitState(c, Seq())).circuit
    val m = inferredC.modules.find(_.name == "m").get

    val syms = SymbolTable.scanModule(m, new LocalSymbolTable with WithMap)
    // The node type is now known
    assert(syms("a").tpe == ir.UIntType(ir.IntWidth(2)) && syms("a").kind == firrtl.NodeKind)
    // The type of the instance is now known because it has been filled in by InferTypes.
    val iType = ir.BundleType(Seq(ir.Field("x", ir.Flip, ir.UIntType(ir.IntWidth(2)))))
    assert(syms("i").tpe == iType && syms("i").kind == firrtl.InstanceKind)
  }

  behavior.of("WithSeq")

  it should "preserve declaration order" in {
    val c = firrtl.Parser.parse(src)
    val m = c.modules.find(_.name == "m").get

    val syms = SymbolTable.scanModule(m, new LocalSymbolTable with WithSeq)
    assert(syms.getSymbols.map(_.name) == Seq("clk", "x", "y", "z", "a", "i", "r", "m"))
  }

  behavior.of("ModuleTypesSymbolTable")

  it should "derive the module type from the module types map" in {
    val c = firrtl.Parser.parse(src)
    val m = c.modules.find(_.name == "m").get

    val childType = ir.BundleType(Seq(ir.Field("x", ir.Flip, ir.UIntType(ir.IntWidth(2)))))
    val moduleTypes = Map("child" -> childType)

    val syms = SymbolTable.scanModule(m, new ModuleTypesSymbolTable(moduleTypes) with WithMap)
    assert(syms.size == 8)
    assert(syms("i").tpe == childType && syms("i").kind == firrtl.InstanceKind)
  }
}
