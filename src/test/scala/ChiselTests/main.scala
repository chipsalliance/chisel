package ChiselTests
import Chisel._
import Chisel.testers._

object MiniChisel {
  def main(gargs: Array[String]): Unit = {
    if (gargs.length < 1)
      println("Need an argument")
    val name   = gargs(0)
    val margs  = Array("--targetDir", "generated")
    val args   = margs ++ gargs.slice(1, gargs.length) 
    name match {
      case "EnableShiftRegister" => chiselMainTest(args, () => Module(new EnableShiftRegister))(c => new EnableShiftRegisterTester(c))
      case "MemorySearch" => chiselMainTest(args, () => Module(new MemorySearch))(c => new MemorySearchTester(c))
      case "VecApp" => chiselMainTest(args, () => Module(new VecApp(4,8)))(c => new VecAppTester(c))
      case "Counter" => chiselMainTest(args, () => Module(new Counter))(c => new CounterTester(c))
      case "Tbl" => chiselMainTest(args, () => Module(new Tbl))(c => new TblTester(c))
      case "LFSR16" => chiselMainTest(args, () => Module(new LFSR16))(c => new LFSR16Tester(c))
      case "Mul" => chiselMainTest(args, () => Module(new Mul(2)))(c => new MulTester(c))
      case "Decoder" => chiselMainTest(args, () => Module(new Decoder))(c => new DecoderTester(c))
      case "VecShiftRegister" => chiselMainTest(args, () => Module(new VecShiftRegister))(c => new VecShiftRegisterTester(c))
      case "RegisterVecShift" => chiselMainTest(args, () => Module(new RegisterVecShift))(c => new RegisterVecShiftTester(c))
      case "ModuleVec" => chiselMainTest(args, () => Module(new ModuleVec(2)))(c => new ModuleVecTester(c))
      case "ModuleWire" => chiselMainTest(args, () => Module(new ModuleWire))(c => new ModuleWireTester(c))
      case "BundleWire" => chiselMainTest(args, () => Module(new BundleWire))(c => new BundleWireTester(c))

      case "Stack" => chiselMainTest(args, () => Module(new Stack(16)))(c => new StackTester(c))
      case "GCD" => chiselMainTest(args, () => Module(new GCD))(c => new GCDTester(c))
      case "Risc" => chiselMainTest(args, () => Module(new Risc))(c => new RiscTester(c))
      case "Rom" => chiselMainTest(args, () => Module(new Rom))(c => new RomTester(c))
      case "Outer" => chiselMainTest(args, () => Module(new Outer))(c => new OuterTester(c))
      case "ComplexAssign" => chiselMainTest(args, () => Module(new ComplexAssign(10)))(c => new ComplexAssignTester(c))
      case "UIntOps" => chiselMainTest(args, () => Module(new UIntOps))(c => new UIntOpsTester(c))
      case "SIntOps" => chiselMainTest(args, () => Module(new SIntOps))(c => new SIntOpsTester(c))
      case "BitsOps" => chiselMainTest(args, () => Module(new BitsOps))(c => new BitsOpsTester(c))
      case "DirChange" => chiselMainTest(args, () => Module(new DirChange))(c => new DirChangeTester(c))
      case "VendingMachine" => chiselMainTest(args, () => Module(new VendingMachine))(c => new VendingMachineTester(c))
      case "Pads" => chiselMainTest(args, () => Module(new Pads))(c => new PadsTester(c))
    }
  }
}
