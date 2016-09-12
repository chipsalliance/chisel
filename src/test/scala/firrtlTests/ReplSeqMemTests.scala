package firrtlTests

import firrtl._
import firrtl.passes._
import Annotations._

class ReplSeqMemSpec extends SimpleTransformSpec {
  val passSeq = Seq(
    ConstProp, CommonSubexpressionElimination, DeadCodeElimination, RemoveEmpty)
  def transforms (writer: java.io.Writer) = Seq(
    new Chisel3ToHighFirrtl(),
    new IRToWorkingIR(),
    new ResolveAndCheck(),
    new HighFirrtlToMiddleFirrtl(),
    new passes.InferReadWrite(TransID(-1)),
    new passes.ReplSeqMem(TransID(-2)),
    new MiddleFirrtlToLowFirrtl(),
    (new Transform with SimpleRun {
     def execute(c: ir.Circuit, a: AnnotationMap) = run(c, passSeq) } ),
    new EmitFirrtl(writer)
  )

  "ReplSeqMem" should "generate blackbox wrappers" in {
    val input = """
circuit Top : 
  module Top : 
    input clk : Clock
    input reset : UInt<1>
    input head_ptr : UInt<5>
    input tail_ptr : UInt<5>
    input wmask : {takens : UInt<2>, history : UInt<14>, info : UInt<14>}
    output io : {backend : {flip allocate : {valid : UInt<1>, bits : {info : {takens : UInt<2>, history : UInt<14>, info : UInt<14>}}}}, commit_entry : {valid : UInt<1>, bits : {info : {takens : UInt<2>, history : UInt<14>, info : UInt<14>}}}}
    output io2 : {backend : {flip allocate : {valid : UInt<1>, bits : {info : {takens : UInt<2>, history : UInt<14>, info : UInt<14>}}}}, commit_entry : {valid : UInt<1>, bits : {info : {takens : UInt<2>, history : UInt<14>, info : UInt<14>}}}}

    io is invalid
    io2 is invalid

    smem entries_info : {takens : UInt<2>, history : UInt<14>, info : UInt<14>}[24]
    when io.backend.allocate.valid :
      write mport W = entries_info[tail_ptr], clk
      W <- io.backend.allocate.bits.info

    read mport R = entries_info[head_ptr], clk
    io.commit_entry.bits.info <- R

    smem entries_info2 : {takens : UInt<2>, history : UInt<14>, info : UInt<14>}[24]
    when io2.backend.allocate.valid :
      write mport W1 = entries_info2[tail_ptr], clk
      when wmask.takens :
        W1.takens <- io.backend.allocate.bits.info.takens
      when wmask.history :
        W1.history <- io.backend.allocate.bits.info.history
      when wmask.info :
        W1.info <- io.backend.allocate.bits.info.history
      
    read mport R1 = entries_info2[head_ptr], clk
    io2.commit_entry.bits.info <- R1
""".stripMargin
    val confLoc = "ReplSeqMemTests.confTEMP"
    val aMap = AnnotationMap(Seq(ReplSeqMemAnnotation("-c:Top:-o:"+confLoc, TransID(-2))))
    val writer = new java.io.StringWriter
    compile(parse(input), aMap, writer)
    // Check correctness of firrtl
    parse(writer.toString)
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem Utility -- getConnectOrigin" should 
      "determine connect origin across nodes/PrimOps even if ConstProp isn't performed" in {
    def checkConnectOrigin(hurdle: String, origin: String) = {
      val input = s"""
circuit Top :
  module Top :
    input a: UInt<1>
    input b: UInt<1>
    input e: UInt<1>
    output c: UInt<1>
    output f: UInt<1>
    node d = $hurdle
    c <= d
    f <= c
""".stripMargin

      val circuit = InferTypes.run(ToWorkingIR.run(parse(input)))
      val m = circuit.modules.head.asInstanceOf[ir.Module]
      val connects = AnalysisUtils.getConnects(m)
      val calculatedOrigin = AnalysisUtils.getConnectOrigin(connects,"f").serialize 
      require(calculatedOrigin == origin, s"getConnectOrigin returns incorrect origin $calculatedOrigin !")
    }

    val tests = List(
      """mux(a, UInt<1>("h1"), UInt<1>("h0"))""" -> "a",
      """mux(UInt<1>("h1"), a, b)""" -> "a",
      """mux(UInt<1>("h0"), a, b)""" -> "b",
      "mux(b, a, a)" -> "a",
      """mux(a, a, UInt<1>("h0"))""" -> "a",
      "mux(a, b, e)" -> "mux(a, b, e)",
      """or(a, UInt<1>("h1"))""" -> """UInt<1>("h1")""",
      """and(a, UInt<1>("h0"))""" -> """UInt<1>("h0")""",
      """UInt<1>("h1")""" -> """UInt<1>("h1")""",
      "asUInt(a)" -> "a",
      "asSInt(a)" -> "a",
      "asClock(a)" -> "a",
      "a" -> "a",
      "or(a, b)" -> "or(a, b)",
      "bits(a, 0, 0)" -> "a"
    )

    tests foreach { case(hurdle, origin) => checkConnectOrigin(hurdle, origin) }

  }
}

// TODO: make more checks
// readwrite vs. no readwrite
// redundant memories (multiple instances of the same type of memory)
// mask + no mask
// conf
