// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.passes._
import firrtl.transforms._
import firrtl.passes.memlib._
import annotations._
import FirrtlCheckers._

class ReplSeqMemSpec extends SimpleTransformSpec {
  def emitter = new LowFirrtlEmitter
  def transforms = Seq(
    new ChirrtlToHighFirrtl(),
    new IRToWorkingIR(),
    new ResolveAndCheck(),
    new HighFirrtlToMiddleFirrtl(),
    new InferReadWrite(),
    new ReplSeqMem(),
    new MiddleFirrtlToLowFirrtl(),
    new SeqTransform {
      def inputForm = LowForm
      def outputForm = LowForm
      def transforms = Seq(new ConstantPropagation, CommonSubexpressionElimination, new DeadCodeElimination, RemoveEmpty)
    }
  )

  "ReplSeqMem" should "generate blackbox wrappers for mems of bundle type" in {
    val input = """
circuit Top : 
  module Top : 
    input clock : Clock
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
      write mport W = entries_info[tail_ptr], clock
      W <- io.backend.allocate.bits.info

    read mport R = entries_info[head_ptr], clock
    io.commit_entry.bits.info <- R

    smem entries_info2 : {takens : UInt<2>, history : UInt<14>, info : UInt<14>}[24]
    when io2.backend.allocate.valid :
      write mport W1 = entries_info2[tail_ptr], clock
      when wmask.takens :
        W1.takens <- io.backend.allocate.bits.info.takens
      when wmask.history :
        W1.history <- io.backend.allocate.bits.info.history
      when wmask.info :
        W1.info <- io.backend.allocate.bits.info.history
      
    read mport R1 = entries_info2[head_ptr], clock
    io2.commit_entry.bits.info <- R1
""".stripMargin
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(ReplSeqMemAnnotation.parse("-c:Top:-o:"+confLoc))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    parse(res.getEmittedCircuit.value)
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem" should "not infinite loop if control signals are derived from registered versions of themselves" in {
    val input = """
circuit Top :
  module Top :
    input clock : Clock
    input hsel : UInt<1>

    reg p_valid : UInt<1>, clock
    reg p_address : UInt<5>, clock
    smem mem : UInt<8>[8][32] 
    when hsel : 
      when p_valid : 
        write mport T_155 = mem[p_address], clock
""".stripMargin
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(ReplSeqMemAnnotation.parse("-c:Top:-o:"+confLoc))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    parse(res.getEmittedCircuit.value)
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem" should "not fail with FixedPoint types " in {
    val input = """
circuit CustomMemory : 
  module CustomMemory : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip rClk : Clock, flip rAddr : UInt<3>, dO : Fixed<16><<8>>, flip wClk : Clock, flip wAddr : UInt<3>, flip wEn : UInt<1>, flip dI : Fixed<16><<8>>}
    
    io is invalid
    smem mem : Fixed<16><<8>>[7] 
    read mport _T_17 = mem[io.rAddr], clock
    io.dO <= _T_17 
    when io.wEn : 
      write mport _T_18 = mem[io.wAddr], clock
      _T_18 <= io.dI
      skip 
""".stripMargin
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(ReplSeqMemAnnotation.parse("-c:CustomMemory:-o:"+confLoc))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    parse(res.getEmittedCircuit.value)
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem" should "not fail with Signed types " in {
    val input = """
circuit CustomMemory : 
  module CustomMemory : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip rClk : Clock, flip rAddr : UInt<3>, dO : SInt<16>, flip wClk : Clock, flip wAddr : UInt<3>, flip wEn : UInt<1>, flip dI : SInt<16>}
    
    io is invalid
    smem mem : SInt<16>[7] 
    read mport _T_17 = mem[io.rAddr], clock
    io.dO <= _T_17 
    when io.wEn : 
      write mport _T_18 = mem[io.wAddr], clock
      _T_18 <= io.dI
      skip 
""".stripMargin
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(ReplSeqMemAnnotation.parse("-c:CustomMemory:-o:"+confLoc))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    parse(res.getEmittedCircuit.value)
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
      val calculatedOrigin = AnalysisUtils.getOrigin(connects, "f").serialize 
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
      "bits(a, 0, 0)" -> "a",
      "validif(a, b)" -> "b"
    )

    tests foreach { case(hurdle, origin) => checkConnectOrigin(hurdle, origin) }

  }
  "ReplSeqMem" should "not de-duplicate memories with the nodedupe annotation " in {
    val input = """
circuit CustomMemory :
  module CustomMemory :
    input clock : Clock
    input reset : UInt<1>
    output io : {flip rClk : Clock, flip rAddr : UInt<3>, dO : UInt<16>, flip wClk : Clock, flip wAddr : UInt<3>, flip wEn : UInt<1>, flip dI : UInt<16>}

    io is invalid
    smem mem_0 : UInt<16>[7]
    smem mem_1 : UInt<16>[7]
    read mport _T_17 = mem_0[io.rAddr], clock
    read mport _T_19 = mem_1[io.rAddr], clock
    io.dO <= and(_T_17, _T_19)
    when io.wEn :
      write mport _T_18 = mem_0[io.wAddr], clock
      write mport _T_20 = mem_1[io.wAddr], clock
      _T_18 <= io.dI
      _T_20 <= io.dI
      skip
"""
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(
      ReplSeqMemAnnotation.parse("-c:CustomMemory:-o:"+confLoc),
      NoDedupMemAnnotation(ComponentName("mem_0", ModuleName("CustomMemory",CircuitName("CustomMemory")))))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    val circuit = parse(res.getEmittedCircuit.value)
    val numExtMods = circuit.modules.count {
      case e: ExtModule =>  true
      case _ => false
    }
    numExtMods should be (2)
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem" should "only not de-duplicate memories with the nodedupe annotation " in {
    val input = """
circuit CustomMemory :
  module CustomMemory :
    input clock : Clock
    input reset : UInt<1>
    output io : {flip rClk : Clock, flip rAddr : UInt<3>, dO : UInt<16>, flip wClk : Clock, flip wAddr : UInt<3>, flip wEn : UInt<1>, flip dI : UInt<16>}

    io is invalid
    smem mem_0 : UInt<16>[7]
    smem mem_1 : UInt<16>[7]
    smem mem_2 : UInt<16>[7]
    read mport _T_17 = mem_0[io.rAddr], clock
    read mport _T_19 = mem_1[io.rAddr], clock
    read mport _T_21 = mem_2[io.rAddr], clock
    io.dO <= and(_T_17, and(_T_19, _T_21))
    when io.wEn :
      write mport _T_18 = mem_0[io.wAddr], clock
      write mport _T_20 = mem_1[io.wAddr], clock
      write mport _T_22 = mem_2[io.wAddr], clock
      _T_18 <= io.dI
      _T_20 <= io.dI
      _T_22 <= io.dI
      skip
"""
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(
      ReplSeqMemAnnotation.parse("-c:CustomMemory:-o:"+confLoc),
      NoDedupMemAnnotation(ComponentName("mem_1", ModuleName("CustomMemory",CircuitName("CustomMemory")))))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    val circuit = parse(res.getEmittedCircuit.value)
    val numExtMods = circuit.modules.count {
      case e: ExtModule =>  true
      case _ => false
    }
    numExtMods should be (2)
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem" should "dedup mems with the same instance name as other mems (in other modules) marked NoDedup" in {
    val input = """
circuit CustomMemory :
  module ChildMemory :
    input clock : Clock
    input reset : UInt<1>
    output io : {flip rClk : Clock, flip rAddr : UInt<3>, dO : UInt<16>, flip wClk : Clock, flip wAddr : UInt<3>, flip wEn : UInt<1>, flip dI : UInt<16>}

    smem mem_0 : UInt<16>[7]
    read mport r1 = mem_0[io.rAddr], clock
    io.dO <= r1
    when io.wEn :
      write mport w1 = mem_0[io.wAddr], clock
      w1 <= io.dI

  module CustomMemory :
    input clock : Clock
    input reset : UInt<1>
    output io : {flip rClk : Clock, flip rAddr : UInt<3>, dO : UInt<16>, flip wClk : Clock, flip wAddr : UInt<3>, flip wEn : UInt<1>, flip dI : UInt<16>}

    inst child of ChildMemory
    child.clock <= clock
    child.reset <= reset
    io <- child.io

    smem mem_0 : UInt<16>[7]
    smem mem_1 : UInt<16>[7]
    read mport r1 = mem_0[io.rAddr], clock
    read mport r2 = mem_1[io.rAddr], clock
    io.dO <= and(r1, and(r2, child.io.dO))
    when io.wEn :
      write mport w1 = mem_0[io.wAddr], clock
      write mport w2 = mem_1[io.wAddr], clock
      w1 <= io.dI
      w2 <= io.dI
"""
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(
      ReplSeqMemAnnotation.parse("-c:CustomMemory:-o:"+confLoc),
      NoDedupMemAnnotation(ComponentName("mem_0", ModuleName("ChildMemory",CircuitName("CustomMemory")))))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    val circuit = parse(res.getEmittedCircuit.value)
    val numExtMods = circuit.modules.count {
      case e: ExtModule =>  true
      case _ => false
    }
    // Note that there are 3 identical SeqMems in this test
    // If the NoDedupMemAnnotation were ignored, we'd end up with just 1 ExtModule
    // If the NoDedupMemAnnotation were handled incorrectly as it was prior to this test, there
    //   would be 3 ExtModules
    numExtMods should be (2)
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem" should "de-duplicate memories without an annotation " in {
    val input = """
circuit CustomMemory :
  module CustomMemory :
    input clock : Clock
    input reset : UInt<1>
    output io : {flip rClk : Clock, flip rAddr : UInt<3>, dO : UInt<16>, flip wClk : Clock, flip wAddr : UInt<3>, flip wEn : UInt<1>, flip dI : UInt<16>}

    io is invalid
    smem mem_0 : UInt<16>[7]
    smem mem_1 : UInt<16>[7]
    read mport _T_17 = mem_0[io.rAddr], clock
    read mport _T_19 = mem_1[io.rAddr], clock
    io.dO <= _T_17
    when io.wEn :
      write mport _T_18 = mem_0[io.wAddr], clock
      write mport _T_20 = mem_1[io.wAddr], clock
      _T_18 <= io.dI
      _T_20 <= io.dI
      skip
"""
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(ReplSeqMemAnnotation.parse("-c:CustomMemory:-o:"+confLoc))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    val circuit = parse(res.getEmittedCircuit.value)
    val numExtMods = circuit.modules.count {
      case e: ExtModule =>  true
      case _ => false
    }
    require(numExtMods == 1)
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem" should "should not have a mask if there is none" in {
    val input = """
circuit CustomMemory :
  module CustomMemory :
    input clock : Clock
    output io : { flip en : UInt<1>, out : UInt<8>[2], flip raddr : UInt<10>, flip waddr : UInt<10>, flip wdata : UInt<8>[2] }

    smem mem : UInt<8>[2][1024]
    read mport r = mem[io.raddr], clock
    io.out <= r

    when io.en :
      write mport w = mem[io.waddr], clock
      w <= io.wdata
"""
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(ReplSeqMemAnnotation.parse("-c:CustomMemory:-o:"+confLoc))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    res.getEmittedCircuit.value shouldNot include ("mask")
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem" should "should not conjoin enable signal with mask condition" in {
    val input = """
circuit CustomMemory :
  module CustomMemory :
    input clock : Clock
    output io : { flip en : UInt<1>, out : UInt<8>[2], flip raddr : UInt<10>, flip waddr : UInt<10>, flip wdata : UInt<8>[2], flip mask : UInt<8>[2] }

    smem mem : UInt<8>[2][1024]
    read mport r = mem[io.raddr], clock
    io.out <= r

    when io.en :
      write mport w = mem[io.waddr], clock
      when io.mask[0] :
        w[0] <= io.wdata[0]
      when io.mask[1] :
        w[1] <= io.wdata[1]
"""
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(ReplSeqMemAnnotation.parse("-c:CustomMemory:-o:"+confLoc))
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // TODO Until RemoveCHIRRTL is removed, enable will still drive validif for mask
    res should containLine ("mem.W0_mask_0 <= validif(io_en, io_mask_0)")
    res should containLine ("mem.W0_mask_1 <= validif(io_en, io_mask_1)")
    (new java.io.File(confLoc)).delete()
  }

  "ReplSeqMem" should "should not conjoin enable signal with wmask condition (RW Port)" in {
    val input = """
circuit CustomMemory :
  module CustomMemory :
    input clock : Clock
    output io : { flip en : UInt<1>, out : UInt<8>[2], flip raddr : UInt<10>, flip waddr : UInt<10>, flip wdata : UInt<8>[2], flip mask : UInt<8>[2] }

    io.out is invalid

    smem mem : UInt<8>[2][1024]

    when io.en :
      write mport w = mem[io.waddr], clock
      when io.mask[0] :
        w[0] <= io.wdata[0]
      when io.mask[1] :
        w[1] <= io.wdata[1]
    when not(io.en) :
      read mport r = mem[io.raddr], clock
      io.out <= r

"""
    val confLoc = "ReplSeqMemTests.confTEMP"
    val annos = Seq(ReplSeqMemAnnotation.parse("-c:CustomMemory:-o:"+confLoc),
                    InferReadWriteAnnotation)
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // TODO Until RemoveCHIRRTL is removed, enable will still drive validif for mask
    res should containLine ("mem.RW0_wmask_0 <= validif(io_en, io_mask_0)")
    res should containLine ("mem.RW0_wmask_1 <= validif(io_en, io_mask_1)")
    (new java.io.File(confLoc)).delete()
  }
}

// TODO: make more checks
// conf
