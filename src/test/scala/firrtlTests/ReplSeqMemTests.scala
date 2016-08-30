package firrtlTests

import firrtl._
import firrtl.passes._
import Annotations._

class ReplSeqMemSpec extends SimpleTransformSpec {

  def transforms (writer: java.io.Writer) = Seq(
    new Chisel3ToHighFirrtl(),
    new IRToWorkingIR(),
    new ResolveAndCheck(),
    new HighFirrtlToMiddleFirrtl(),
    new passes.InferReadWrite(TransID(-1)),
    new passes.ReplSeqMem(TransID(-2)),
    new MiddleFirrtlToLowFirrtl(),
    new EmitFirrtl(writer)
  )

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

    tests.foreach{ case(hurdle, origin) => checkConnectOrigin(hurdle, origin) }

  }

  "ReplSeqMem" should "generate blackbox wrappers (no wmask, r, w ports)" in {
    val input = """
circuit sram6t :
  module sram6t :
    input clk : Clock
    input reset : UInt<1>
    output io : {flip en : UInt<1>, flip wen : UInt<1>, flip waddr : UInt<8>, flip wdata : UInt<32>, flip raddr : UInt<8>, rdata : UInt<32>}

    io is invalid
    smem mem : UInt<32>[128]
    node T_0 = eq(io.wen, UInt<1>("h00"))
    node T_1 = and(io.en, T_0)
    wire T_2 : UInt
    T_2 is invalid
    when T_1 :
      T_2 <= io.raddr
    read mport T_3 = mem[T_2], clk
    io.rdata <= T_3
    node T_4 = and(io.en, io.wen)
    when T_4 :
      write mport T_5 = mem[io.waddr], clk
      T_5 <= io.wdata
""".stripMargin

    val check = """
circuit sram6t :
  module sram6t :
    input clk : Clock
    input reset : UInt<1>
    input io_en : UInt<1>
    input io_wen : UInt<1>
    input io_waddr : UInt<8>
    input io_wdata : UInt<32>
    input io_raddr : UInt<8>
    output io_rdata : UInt<32>
  
    inst mem of mem
    node T_0 = eq(io_wen, UInt<1>("h0"))
    node T_1 = and(io_en, T_0)
    wire T_2 : UInt<8>
    node GEN_0 = validif(T_1, io_raddr)
    node GEN_1 = mux(T_1, UInt<1>("h1"), UInt<1>("h0"))
    node T_4 = and(io_en, io_wen)
    node GEN_2 = validif(T_4, io_waddr)
    node GEN_3 = validif(T_4, clk)
    node GEN_4 = mux(T_4, UInt<1>("h1"), UInt<1>("h0"))
    node GEN_5 = validif(T_4, io_wdata)
    node GEN_6 = mux(T_4, UInt<1>("h1"), UInt<1>("h0"))
    io_rdata <= mem.R0_data
    mem.R0_addr <= bits(T_2, 6, 0)
    mem.R0_clk <= clk
    mem.R0_en <= GEN_1
    mem.W0_addr <= bits(GEN_2, 6, 0)
    mem.W0_clk <= GEN_3
    mem.W0_en <= GEN_4
    mem.W0_data <= GEN_5
    T_2 <= GEN_0

  extmodule mem_ext :
    input R0_addr : UInt<7>
    input R0_en : UInt<1>
    input R0_clk : Clock
    output R0_data : UInt<32>
    input W0_addr : UInt<7>
    input W0_en : UInt<1>
    input W0_clk : Clock
    input W0_data : UInt<32>
  

  module mem :
    input R0_addr : UInt<7>
    input R0_en : UInt<1>
    input R0_clk : Clock
    output R0_data : UInt<32>
    input W0_addr : UInt<7>
    input W0_en : UInt<1>
    input W0_clk : Clock
    input W0_data : UInt<32>
  
    inst mem_ext of mem_ext
    mem_ext.R0_addr <= R0_addr
    mem_ext.R0_en <= R0_en
    mem_ext.R0_clk <= R0_clk
    R0_data <= bits(mem_ext.R0_data, 31, 0)
    mem_ext.W0_addr <= W0_addr
    mem_ext.W0_en <= W0_en
    mem_ext.W0_clk <= W0_clk
    mem_ext.W0_data <= W0_data     
""".stripMargin

    val checkConf = """name mem_ext depth 128 width 32 ports write,read  """
    
    def read(file: String) = scala.io.Source.fromFile(file).getLines.mkString("\n")
    
    val confLoc = "ReplSeqMemTests.confTEMP"
    val aMap = AnnotationMap(Seq(ReplSeqMemAnnotation("-c:sram6t:-o:"+confLoc, TransID(-2))))
    val writer = new java.io.StringWriter
    execute(writer, aMap, input, check)
    val confOut = read(confLoc)
    require(confOut==checkConf, "Conf file incorrect!")
    (new java.io.File(confLoc)).delete()
  }
}

// TODO: make more checks
// readwrite vs. no readwrite
// redundant memories (multiple instances of the same type of memory)
// mask + no mask
// conf