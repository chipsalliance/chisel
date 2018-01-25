// See LICENSE for license details.

package chiselTests

import collection.immutable.ListMap

// Keep Chisel._ separate from chisel3._ below
object CompatibilityComponents {
  import Chisel._
  import Chisel3Components._

  class ChiselBundle extends Bundle {
    val a = UInt(width = 32)
    val b = UInt(width = 32).flip

    override def cloneType = (new ChiselBundle).asInstanceOf[this.type]
  }
  class ChiselRecord extends Record {
    val elements = ListMap("a" -> UInt(width = 32), "b" -> UInt(width = 32).flip)
    override def cloneType = (new ChiselRecord).asInstanceOf[this.type]
  }

  abstract class ChiselDriverModule(_io: => Record) extends Module {
    val io = _io
    io.elements("a").asUInt := UInt(123)
    assert(io.elements("b").asUInt === UInt(123))
  }
  abstract class ChiselPassthroughModule(_io: => Record) extends Module {
    val io = _io
    io.elements("b").asUInt := io.elements("a").asUInt
  }

  class ChiselBundleModuleA extends ChiselDriverModule(new ChiselBundle)
  class ChiselBundleModuleB extends ChiselPassthroughModule((new ChiselBundle).flip)
  class ChiselRecordModuleA extends ChiselDriverModule(new ChiselRecord)
  class ChiselRecordModuleB extends ChiselPassthroughModule((new ChiselRecord).flip)

  class ChiselModuleChisel3BundleA extends ChiselDriverModule(new Chisel3Bundle)
  class ChiselModuleChisel3BundleB extends ChiselPassthroughModule((new Chisel3Bundle).flip)
  class ChiselModuleChisel3RecordA extends ChiselDriverModule(new Chisel3Record)
  class ChiselModuleChisel3RecordB extends ChiselPassthroughModule((new Chisel3Record).flip)
}

object Chisel3Components {
  import chisel3._
  import CompatibilityComponents._

  class Chisel3Bundle extends Bundle {
    val a = Output(UInt(32.W))
    val b = Input(UInt(32.W))

    override def cloneType = (new Chisel3Bundle).asInstanceOf[this.type]
  }

  class Chisel3Record extends Record {
    val elements = ListMap("a" -> Output(UInt(32.W)), "b" -> Input(UInt(32.W)))
    override def cloneType = (new Chisel3Record).asInstanceOf[this.type]
  }

  abstract class Chisel3DriverModule(_io: => Record) extends Module {
    val io = IO(_io)
    io.elements("a").asUInt := 123.U
    assert(io.elements("b").asUInt === 123.U)
  }
  abstract class Chisel3PassthroughModule(_io: => Record) extends Module {
    val io = IO(_io)
    io.elements("b").asUInt := io.elements("a").asUInt
  }

  class Chisel3BundleModuleA extends Chisel3DriverModule(new Chisel3Bundle)
  class Chisel3BundleModuleB extends Chisel3PassthroughModule(Flipped(new Chisel3Bundle))
  class Chisel3RecordModuleA extends Chisel3DriverModule(new Chisel3Record)
  class Chisel3RecordModuleB extends Chisel3PassthroughModule(Flipped(new Chisel3Record))

  class Chisel3ModuleChiselBundleA extends Chisel3DriverModule(new ChiselBundle)
  class Chisel3ModuleChiselBundleB extends Chisel3PassthroughModule(Flipped(new ChiselBundle))
  class Chisel3ModuleChiselRecordA extends Chisel3DriverModule(new ChiselRecord)
  class Chisel3ModuleChiselRecordB extends Chisel3PassthroughModule(Flipped(new ChiselRecord))
}

class CompatibiltyInteroperabilitySpec extends ChiselFlatSpec {

  "Modules defined in the Chisel._" should "successfully bulk connect in chisel3._" in {
    import chisel3._
    import chisel3.testers.BasicTester
    import CompatibilityComponents._

    assertTesterPasses(new BasicTester {
      val a = Module(new ChiselBundleModuleA)
      val b = Module(new ChiselBundleModuleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new ChiselRecordModuleA)
      val b = Module(new ChiselRecordModuleB)
      b.io <> a.io
      stop()
    })
  }

  "Moduless defined in the chisel3._" should "successfully bulk connect in Chisel._" in {
    import Chisel._
    import chisel3.testers.BasicTester
    import Chisel3Components._

    assertTesterPasses(new BasicTester {
      val a = Module(new Chisel3BundleModuleA)
      val b = Module(new Chisel3BundleModuleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new Chisel3RecordModuleA)
      val b = Module(new Chisel3RecordModuleB)
      b.io <> a.io
      stop()
    })
  }


  "Bundles defined in Chisel._" should "work in chisel3._ Modules" in {
    import chisel3._
    import chisel3.testers.BasicTester
    import Chisel3Components._

    assertTesterPasses(new BasicTester {
      val a = Module(new Chisel3ModuleChiselBundleA)
      val b = Module(new Chisel3ModuleChiselBundleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new Chisel3ModuleChiselRecordA)
      val b = Module(new Chisel3ModuleChiselRecordB)
      b.io <> a.io
      stop()
    })
  }

  "Bundles defined in chisel3._" should "work in Chisel._ Modules" in {
    import chisel3._
    import chisel3.testers.BasicTester
    import CompatibilityComponents._

    assertTesterPasses(new BasicTester {
      val a = Module(new ChiselModuleChisel3BundleA)
      val b = Module(new ChiselModuleChisel3BundleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new ChiselModuleChisel3RecordA)
      val b = Module(new ChiselModuleChisel3RecordB)
      b.io <> a.io
      stop()
    })
  }


  "Similar Bundles defined in the chisel3._ and Chisel._" should
      "successfully bulk connect in chisel3._" in {
    import chisel3._
    import chisel3.testers.BasicTester
    import Chisel3Components._
    import CompatibilityComponents._

    assertTesterPasses(new BasicTester {
      val a = Module(new ChiselBundleModuleA)
      val b = Module(new Chisel3BundleModuleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new Chisel3BundleModuleA)
      val b = Module(new ChiselBundleModuleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new ChiselRecordModuleA)
      val b = Module(new Chisel3RecordModuleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new Chisel3RecordModuleA)
      val b = Module(new ChiselRecordModuleB)
      b.io <> a.io
      stop()
    })
  }
  they should "successfully bulk connect in Chisel._" in {
    import Chisel._
    import chisel3.testers.BasicTester
    import Chisel3Components._
    import CompatibilityComponents._

    assertTesterPasses(new BasicTester {
      val a = Module(new ChiselBundleModuleA)
      val b = Module(new Chisel3BundleModuleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new Chisel3BundleModuleA)
      val b = Module(new ChiselBundleModuleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new ChiselRecordModuleA)
      val b = Module(new Chisel3RecordModuleB)
      b.io <> a.io
      stop()
    })
    assertTesterPasses(new BasicTester {
      val a = Module(new Chisel3RecordModuleA)
      val b = Module(new ChiselRecordModuleB)
      b.io <> a.io
      stop()
    })
  }

  "An instance of a chisel3.Module inside a Chisel.Module" should "have its inputs invalidated" in {
    compile {
      import Chisel._
      new Module {
        val io = new Bundle {
          val in = UInt(INPUT, width = 32)
          val cond = Bool(INPUT)
          val out = UInt(OUTPUT, width = 32)
        }
        val children = Seq(Module(new PassthroughModule),
                           Module(new PassthroughMultiIOModule),
                           Module(new PassthroughRawModule))
        io.out := children.map(_.io.out).reduce(_ + _)
        children.foreach { child =>
          when (io.cond) {
            child.io.in := io.in
          }
        }
      }
    }
  }
}

