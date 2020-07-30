// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage

class BundleWithIntArg(val i: Int) extends Bundle {
  val out = UInt(i.W)
}

class BundleWithImplicit()(implicit val ii: Int) extends Bundle {
  val out = UInt(ii.W)
}

class BundleWithArgAndImplicit(val i: Int)(implicit val ii: Int) extends Bundle {
  val out1 = UInt(i.W)
  val out2 = UInt(ii.W)
}

class BaseBundleVal(val i: Int) extends Bundle {
  val inner = UInt(i.W)
}
class SubBundle(i: Int, val i2: Int) extends BaseBundleVal(i) {
  val inner2 = UInt(i2.W)
}
class SubBundleInvalid(i: Int, val i2: Int) extends BaseBundleVal(i + 1) {
  val inner2 = UInt(i2.W)
}

class BaseBundleNonVal(i: Int) extends Bundle {
  val inner = UInt(i.W)
}
class SubBundleVal(val i: Int, val i2: Int) extends BaseBundleNonVal(i) {
  val inner2 = UInt(i2.W)
}

class ModuleWithInner extends Module {
  class InnerBundle(val i: Int) extends Bundle {
    val out = UInt(i.W)
  }

  val io = IO(new Bundle{})

  val myWire = Wire(new InnerBundle(14))
  require(myWire.i == 14)
}

object CompanionObjectWithBundle {
  class ParameterizedInner(val i: Int) extends Bundle {
    val data = UInt(i.W)
  }
  class Inner extends Bundle {
    val data = UInt(8.W)
  }
}

class NestedAnonymousBundle extends Bundle {
  val a = Output(new Bundle {
    val a = UInt(8.W)
  })
}

// A Bundle with an argument that is also a field.
// Not necessarily good style (and not necessarily recommended), but allowed to preserve compatibility.
class BundleWithArgumentField(val x: Data, val y: Data) extends Bundle

class AutoClonetypeSpec extends ChiselFlatSpec with Utils {
  "Bundles with Scala args" should "not need clonetype" in {
    ChiselStage.elaborate { new Module {
      val io = IO(new Bundle{})

      val myWire = Wire(new BundleWithIntArg(8))
      assert(myWire.i == 8)
    } }
  }

  "Bundles with Scala implicit args" should "not need clonetype" in {
    ChiselStage.elaborate { new Module {
      val io = IO(new Bundle{})

      implicit val implicitInt: Int = 4
      val myWire = Wire(new BundleWithImplicit())

      assert(myWire.ii == 4)
    } }
  }

  "Bundles with Scala explicit and impicit args" should "not need clonetype" in {
    ChiselStage.elaborate { new Module {
      val io = IO(new Bundle{})

      implicit val implicitInt: Int = 4
      val myWire = Wire(new BundleWithArgAndImplicit(8))

      assert(myWire.i == 8)
      assert(myWire.ii == 4)
    } }
  }

  "Subtyped Bundles" should "not need clonetype" in {
    ChiselStage.elaborate { new Module {
      val io = IO(new Bundle{})

      val myWire = Wire(new SubBundle(8, 4))

      assert(myWire.i == 8)
      assert(myWire.i2 == 4)
    } }
    ChiselStage.elaborate { new Module {
      val io = IO(new Bundle{})

      val myWire = Wire(new SubBundleVal(8, 4))

      assert(myWire.i == 8)
      assert(myWire.i2 == 4)
    } }
  }

  "Subtyped Bundles that don't clone well" should "be caught" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new Module {
        val io = IO(new Bundle{})
        val myWire = Wire(new SubBundleInvalid(8, 4))
      } }
    }
  }

  "Inner bundles with Scala args" should "not need clonetype" in {
    ChiselStage.elaborate { new ModuleWithInner }
  }

  "Bundles with arguments as fields" should "not need clonetype" in {
    ChiselStage.elaborate { new Module {
      val io = IO(Output(new BundleWithArgumentField(UInt(8.W), UInt(8.W))))
      io.x := 1.U
      io.y := 1.U
    } }
  }

  "Bundles inside companion objects" should "not need clonetype" in {
    ChiselStage.elaborate { new Module {
      val io = IO(Output(new CompanionObjectWithBundle.Inner))
      io.data := 1.U
    } }
  }

  "Parameterized bundles inside companion objects" should "not need clonetype" in {
    ChiselStage.elaborate { new Module {
      val io = IO(Output(new CompanionObjectWithBundle.ParameterizedInner(8)))
      io.data := 1.U
    } }
  }

  "Nested directioned anonymous Bundles" should "not need clonetype" in {
    ChiselStage.elaborate { new Module {
      val io = IO(new NestedAnonymousBundle)
      val a = WireDefault(io)
      io.a.a := 1.U
    } }
  }

  "3.0 null compatibility" should "not need clonetype" in {
    ChiselStage.elaborate { new Module {
      class InnerClassThing {
        def createBundle: Bundle = new Bundle {
          val a = Output(UInt(8.W))
        }
      }
      val io = IO((new InnerClassThing).createBundle)
      val a = WireDefault(io)
    } }
  }

  "Aliased fields" should "be caught" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new Module {
        val bundleFieldType = UInt(8.W)
        val io = IO(Output(new Bundle {
          val a = bundleFieldType
        }))
        io.a := 0.U
      } }
    }
  }

  "Aliased fields from inadequate autoclonetype" should "be caught" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      class BadBundle(val typeTuple: (Data, Int)) extends Bundle {
        val a = typeTuple._1
      }

      ChiselStage.elaborate { new Module {
        val io = IO(Output(new BadBundle(UInt(8.W), 1)))
        io.a := 0.U
      } }
    }
  }

  "Wrapped IO construction without parent reference" should "not fail for autoclonetype" in {
    class TestModule extends MultiIOModule {
      def thunk[T](f: => T): T = f
      val works = thunk(IO(new Bundle {
        val x = Output(UInt(3.W))
      }))
    }
    ChiselStage.elaborate { new TestModule }
  }

  "Wrapped IO construction with parent references" should "not fail for autoclonetype" in {
    class TestModule(blah: Int) extends MultiIOModule {
      // Note that this currently fails only if f: =>T on Scala 2.11.12
      // This works successfully with 2.12.11
      def thunk[T](f: => T): T = f
      val broken = thunk(IO(new Bundle {
        val x = Output(UInt(blah.W))
      }))
    }
    ChiselStage.elaborate { new TestModule(3) }
  }
}
