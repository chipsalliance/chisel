package chiselTests

import chisel3._

class SuggestNameWire extends NamedModuleTester {
  val wire = expectName(WireInit(3.U), "foo")
  wire.suggestName("foo")
}

class SuggestNameReg extends NamedModuleTester {
  val reg = expectName(RegInit(3.U), "foo")
  reg.suggestName("foo")
}

class SuggestNameNode extends NamedModuleTester {
  val wire = WireInit(3.U)
  val node = expectName(wire + 1.U, "foo")
  node.suggestName("foo")
}

class SuggestNameMem extends NamedModuleTester {
  val mem = expectName(SyncReadMem(32, UInt(8.W)), "foo")
  mem.suggestName("foo")
}

class SuggestNameSpec extends ChiselFlatSpec {

  private def doTest(testMod: => NamedModuleTester): Unit = {
    var module: NamedModuleTester = null
    elaborate { module = testMod; module }
    assert(module.getNameFailures() == Nil)
  }

  behavior of "suggestName"

  it should "name wires" in {
    doTest(new SuggestNameWire)
  }

  it should "name register" in {
    doTest(new SuggestNameReg)
  }

  it should "name nodes" in {
    doTest(new SuggestNameNode)
  }

  it should "name mems" in {
    doTest(new SuggestNameMem)
  }
}
