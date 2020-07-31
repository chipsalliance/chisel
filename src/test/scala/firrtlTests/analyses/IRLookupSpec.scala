package firrtlTests.analyses

import firrtl.PrimOps.AsUInt
import firrtl.analyses.IRLookup
import firrtl.annotations.{CircuitTarget, ModuleTarget, ReferenceTarget}
import firrtl._
import firrtl.ir._
import firrtl.options.Dependency
import firrtl.passes.ExpandWhensAndCheck
import firrtl.stage.{Forms, TransformManager}
import firrtl.testutils.FirrtlFlatSpec


class IRLookupSpec extends FirrtlFlatSpec {

  "IRLookup" should "return declarations" in {
    val input =
      """circuit Test:
        |  module Test :
        |    input in: UInt<8>
        |    input clk: Clock
        |    input reset: UInt<1>
        |    output out: {a: UInt<8>, b: UInt<8>[2]}
        |    input ana1: Analog<8>
        |    output ana2: Analog<8>
        |    out is invalid
        |    reg r: UInt<8>, clk with:
        |      (reset => (reset, UInt(0)))
        |    node x = r
        |    wire y: UInt<8>
        |    y <= x
        |    out.b[0] <= and(y, asUInt(SInt(-1)))
        |    attach(ana1, ana2)
        |    inst child of Child
        |    out.a <= child.out
        |  module Child:
        |    output out: UInt<8>
        |    out <= UInt(1)
        |""".stripMargin

    val circuit = new firrtl.stage.transforms.Compiler(Seq(Dependency[ExpandWhensAndCheck])).runTransform(
      CircuitState(parse(input), UnknownForm)
    ).circuit
    val irLookup = IRLookup(circuit)
    val Test = ModuleTarget("Test", "Test")
    val uint8 = UIntType(IntWidth(8))

    irLookup.declaration(Test.ref("in")) shouldBe Port(NoInfo, "in", Input, uint8)
    irLookup.declaration(Test.ref("clk")) shouldBe Port(NoInfo, "clk", Input, ClockType)
    irLookup.declaration(Test.ref("reset")) shouldBe Port(NoInfo, "reset", Input, UIntType(IntWidth(1)))

    val out = Port(NoInfo, "out", Output,
      BundleType(Seq(Field("a", Default, uint8), Field("b", Default, VectorType(uint8, 2))))
    )
    irLookup.declaration(Test.ref("out")) shouldBe out
    irLookup.declaration(Test.ref("out").field("a")) shouldBe out
    irLookup.declaration(Test.ref("out").field("b").index(0)) shouldBe out
    irLookup.declaration(Test.ref("out").field("b").index(1)) shouldBe out

    irLookup.declaration(Test.ref("ana1")) shouldBe Port(NoInfo, "ana1", Input, AnalogType(IntWidth(8)))
    irLookup.declaration(Test.ref("ana2")) shouldBe Port(NoInfo, "ana2", Output, AnalogType(IntWidth(8)))

    val clk = WRef("clk", ClockType, PortKind, SourceFlow)
    val reset = WRef("reset", UIntType(IntWidth(1)), PortKind, SourceFlow)
    val init = UIntLiteral(0)
    val reg = DefRegister(NoInfo, "r", uint8, clk, reset, init)
    irLookup.declaration(Test.ref("r")) shouldBe reg
    irLookup.declaration(Test.ref("r").clock) shouldBe reg
    irLookup.declaration(Test.ref("r").reset) shouldBe reg
    irLookup.declaration(Test.ref("r").init) shouldBe reg
    irLookup.kindFinder(Test, RegKind) shouldBe Seq(Test.ref("r"))
    irLookup.declaration(Test.ref("x")) shouldBe DefNode(NoInfo, "x", WRef("r", uint8, RegKind, SourceFlow))
    irLookup.declaration(Test.ref("y")) shouldBe DefWire(NoInfo, "y", uint8)

    irLookup.declaration(Test.ref("@and#0")) shouldBe
      DoPrim(PrimOps.And,
        Seq(WRef("y", uint8, WireKind, SourceFlow), DoPrim(AsUInt, Seq(SIntLiteral(-1)), Nil, UIntType(IntWidth(1)))),
        Nil,
        uint8
      )

    val inst = WDefInstance(NoInfo, "child", "Child", BundleType(Seq(Field("out", Default, uint8))))
    irLookup.declaration(Test.ref("child")) shouldBe inst
    irLookup.declaration(Test.ref("child").field("out")) shouldBe inst
    irLookup.declaration(Test.instOf("child", "Child").ref("out")) shouldBe Port(NoInfo, "out", Output, uint8)

    intercept[IllegalArgumentException]{ irLookup.declaration(Test.instOf("child", "Child").ref("missing")) }
    intercept[IllegalArgumentException]{ irLookup.declaration(Test.instOf("child", "Missing").ref("out")) }
    intercept[IllegalArgumentException]{ irLookup.declaration(Test.instOf("missing", "Child").ref("out")) }
    intercept[IllegalArgumentException]{ irLookup.declaration(Test.ref("missing")) }
    intercept[IllegalArgumentException]{ irLookup.declaration(Test.ref("out").field("c")) }
    intercept[IllegalArgumentException]{ irLookup.declaration(Test.instOf("child", "Child").ref("out").field("missing")) }
  }

  "IRLookup" should "return mem declarations" in {
    def commonFields: Seq[String] = Seq("clk", "en", "addr")
    def readerTargets(rt: ReferenceTarget): Seq[ReferenceTarget] = {
      (commonFields ++ Seq("data")).map(rt.field)
    }
    def writerTargets(rt: ReferenceTarget): Seq[ReferenceTarget] = {
      (commonFields ++ Seq("data", "mask")).map(rt.field)
    }
    def readwriterTargets(rt: ReferenceTarget): Seq[ReferenceTarget] = {
      (commonFields ++ Seq("wdata", "wmask", "wmode", "rdata")).map(rt.field)
    }
    val input =
      s"""circuit Test:
         |  module Test :
         |    input in : UInt<8>
         |    input clk: Clock[3]
         |    input dataClk: Clock
         |    input mode: UInt<1>
         |    output out : UInt<8>[2]
         |    mem m:
         |      data-type => UInt<8>
         |      reader => r
         |      writer => w
         |      readwriter => rw
         |      depth => 2
         |      write-latency => 1
         |      read-latency => 0
         |
         |    reg addr: UInt<1>, dataClk
         |    reg en: UInt<1>, dataClk
         |    reg indata: UInt<8>, dataClk
         |
         |    m.r.clk <= clk[0]
         |    m.r.en <= en
         |    m.r.addr <= addr
         |    out[0] <= m.r.data
         |
         |    m.w.clk <= clk[1]
         |    m.w.en <= en
         |    m.w.addr <= addr
         |    m.w.data <= indata
         |    m.w.mask <= en
         |
         |    m.rw.clk <= clk[2]
         |    m.rw.en <= en
         |    m.rw.addr <= addr
         |    m.rw.wdata <= indata
         |    m.rw.wmask <= en
         |    m.rw.wmode <= en
         |    out[1] <= m.rw.rdata
         |""".stripMargin

    val C = CircuitTarget("Test")
    val MemTest = C.module("Test")
    val Mem = MemTest.ref("m")
    val Reader = Mem.field("r")
    val Writer = Mem.field("w")
    val Readwriter = Mem.field("rw")
    val allSignals = readerTargets(Reader) ++ writerTargets(Writer) ++ readwriterTargets(Readwriter)

    val circuit = new firrtl.stage.transforms.Compiler(Seq(Dependency[ExpandWhensAndCheck])).runTransform(
      CircuitState(parse(input), UnknownForm)
    ).circuit
    val irLookup = IRLookup(circuit)
    val uint8 = UIntType(IntWidth(8))
    val mem = DefMemory(NoInfo, "m", uint8, 2, 1, 0, Seq("r"), Seq("w"), Seq("rw"))
    allSignals.foreach { at =>
      irLookup.declaration(at) shouldBe mem
    }
  }

  "IRLookup" should "return expressions, types, kinds, and flows" in {
    val input =
      """circuit Test:
        |  module Test :
        |    input in: UInt<8>
        |    input clk: Clock
        |    input reset: UInt<1>
        |    output out: {a: UInt<8>, b: UInt<8>[2]}
        |    input ana1: Analog<8>
        |    output ana2: Analog<8>
        |    out is invalid
        |    reg r: UInt<8>, clk with:
        |      (reset => (reset, UInt(0)))
        |    node x = r
        |    wire y: UInt<8>
        |    y <= x
        |    out.b[0] <= and(y, asUInt(SInt(-1)))
        |    attach(ana1, ana2)
        |    inst child of Child
        |    out.a <= child.out
        |  module Child:
        |    output out: UInt<8>
        |    out <= UInt(1)
        |""".stripMargin

    val circuit = new firrtl.stage.transforms.Compiler(Seq(Dependency[ExpandWhensAndCheck])).runTransform(
      CircuitState(parse(input), UnknownForm)
    ).circuit
    val irLookup = IRLookup(circuit)
    val Test = ModuleTarget("Test", "Test")
    val uint8 = UIntType(IntWidth(8))
    val uint1 = UIntType(IntWidth(1))

    def check(rt: ReferenceTarget, e: Expression): Unit = {
      irLookup.expr(rt) shouldBe e
      irLookup.tpe(rt) shouldBe e.tpe
      irLookup.kind(rt) shouldBe Utils.kind(e)
      irLookup.flow(rt) shouldBe Utils.flow(e)
    }

    check(Test.ref("in"), WRef("in", uint8, PortKind, SourceFlow))
    check(Test.ref("clk"), WRef("clk", ClockType, PortKind, SourceFlow))
    check(Test.ref("reset"), WRef("reset", uint1, PortKind, SourceFlow))

    val out = Test.ref("out")
    val outExpr =
      WRef("out",
        BundleType(Seq(Field("a", Default, uint8), Field("b", Default, VectorType(uint8, 2)))),
        PortKind,
        SinkFlow
      )
    check(out, outExpr)
    check(out.field("a"), WSubField(outExpr, "a", uint8, SinkFlow))
    val outB = out.field("b")
    val outBExpr = WSubField(outExpr, "b", VectorType(uint8, 2), SinkFlow)
    check(outB, outBExpr)
    check(outB.index(0), WSubIndex(outBExpr, 0, uint8, SinkFlow))
    check(outB.index(1), WSubIndex(outBExpr, 1, uint8, SinkFlow))

    check(Test.ref("ana1"), WRef("ana1", AnalogType(IntWidth(8)), PortKind, SourceFlow))
    check(Test.ref("ana2"), WRef("ana2", AnalogType(IntWidth(8)), PortKind, SinkFlow))

    val clk = WRef("clk", ClockType, PortKind, SourceFlow)
    val reset = WRef("reset", UIntType(IntWidth(1)), PortKind, SourceFlow)
    val init = UIntLiteral(0)
    check(Test.ref("r"), WRef("r", uint8, RegKind, DuplexFlow))
    check(Test.ref("r").clock, clk)
    check(Test.ref("r").reset, reset)
    check(Test.ref("r").init, init)

    check(Test.ref("x"), WRef("x", uint8, ExpKind, SourceFlow))

    check(Test.ref("y"), WRef("y", uint8, WireKind, DuplexFlow))

    check(Test.ref("@and#0"),
      DoPrim(PrimOps.And,
        Seq(WRef("y", uint8, WireKind, SourceFlow), DoPrim(AsUInt, Seq(SIntLiteral(-1)), Nil, UIntType(IntWidth(1)))),
        Nil,
        uint8
      )
    )

    val child = WRef("child", BundleType(Seq(Field("out", Default, uint8))), InstanceKind, SourceFlow)
    check(Test.ref("child"), child)
    check(Test.ref("child").field("out"),
      WSubField(child, "out", uint8, SourceFlow)
    )
  }

  "IRLookup" should "cache expressions" in {
    def mkType(i: Int): String = {
      if(i == 0) "UInt<8>" else s"{x: ${mkType(i - 1)}}"
    }

    val depth = 500

    val input =
      s"""circuit Test:
        |  module Test :
        |    input in: ${mkType(depth)}
        |    output out: ${mkType(depth)}
        |    out <= in
        |""".stripMargin

    val circuit = new firrtl.stage.transforms.Compiler(Seq(Dependency[ExpandWhensAndCheck])).runTransform(
      CircuitState(parse(input), UnknownForm)
    ).circuit
    val Test = ModuleTarget("Test", "Test")
    val irLookup = IRLookup(circuit)
    def mkReferences(parent: ReferenceTarget, i: Int): Seq[ReferenceTarget] = {
      if(i == 0) Seq(parent) else {
        val newParent = parent.field("x")
        newParent +: mkReferences(newParent, i - 1)
      }
    }

    // Check caching from root to leaf
    val inRefs = mkReferences(Test.ref("in"), depth)
    val (inStartTime, _) = Utils.time(irLookup.expr(inRefs.head))
    inRefs.tail.foreach { r =>
      val (ms, _) = Utils.time(irLookup.expr(r))
      require(inStartTime > ms)
    }
    val outRefs = mkReferences(Test.ref("out"), depth).reverse
    val (outStartTime, _) = Utils.time(irLookup.expr(outRefs.head))
    outRefs.tail.foreach { r =>
      val (ms, _) = Utils.time(irLookup.expr(r))
      require(outStartTime > ms)
    }
  }
}
