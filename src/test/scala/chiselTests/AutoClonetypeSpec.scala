// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.TestUtils
import chisel3.util.QueueIO
import chisel3.stage.ChiselStage.elaborate

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

// Needs to be top-level so that reflective autoclonetype works
class InheritingBundle extends QueueIO(UInt(8.W), 8) {
  val error = Output(Bool())
}

class AutoClonetypeSpec extends ChiselFlatSpec with Utils {

  "Bundles with Scala args" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(new Bundle{})

      val myWire = Wire(new BundleWithIntArg(8))
      assert(myWire.i == 8)
    } }
  }

  "Bundles with Scala implicit args" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(new Bundle{})

      implicit val implicitInt: Int = 4
      val myWire = Wire(new BundleWithImplicit())

      assert(myWire.ii == 4)
    } }
  }

  "Bundles with Scala explicit and impicit args" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(new Bundle{})

      implicit val implicitInt: Int = 4
      val myWire = Wire(new BundleWithArgAndImplicit(8))

      assert(myWire.i == 8)
      assert(myWire.ii == 4)
    } }
  }

  "Subtyped Bundles" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(new Bundle{})

      val myWire = Wire(new SubBundle(8, 4))

      assert(myWire.i == 8)
      assert(myWire.i2 == 4)
    } }
    elaborate { new Module {
      val io = IO(new Bundle{})

      val myWire = Wire(new SubBundleVal(8, 4))

      assert(myWire.i == 8)
      assert(myWire.i2 == 4)
    } }
  }

  "Autoclonetype" should "work outside of a builder context" in {
    new BundleWithIntArg(8).cloneType
  }

  "Subtyped Bundles that don't clone well" should "be now be supported!" in {
    elaborate { new Module {
      val io = IO(new Bundle{})
      val myWire = Wire(new SubBundleInvalid(8, 4))
    } }
  }

  "Inner bundles with Scala args" should "not need clonetype" in {
    elaborate { new ModuleWithInner }
  }

  "Bundles with arguments as fields" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(Output(new BundleWithArgumentField(UInt(8.W), UInt(8.W))))
      io.x := 1.U
      io.y := 1.U
    } }
  }

  it should "also work when giving directions to the fields" in {
    elaborate { new Module {
      val io = IO(new BundleWithArgumentField(Input(UInt(8.W)), Output(UInt(8.W))))
      io.y := io.x
    } }
  }

  "Bundles inside companion objects" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(Output(new CompanionObjectWithBundle.Inner))
      io.data := 1.U
    } }
  }

  "Parameterized bundles inside companion objects" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(Output(new CompanionObjectWithBundle.ParameterizedInner(8)))
      io.data := 1.U
    } }
  }

  "Nested directioned anonymous Bundles" should "not need clonetype" in {
    elaborate { new Module {
      val io = IO(new NestedAnonymousBundle)
      val a = WireDefault(io)
      io.a.a := 1.U
    } }
  }

  "3.0 null compatibility" should "not need clonetype" in {
    elaborate { new Module {
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
      elaborate { new Module {
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

      elaborate { new Module {
        val io = IO(Output(new BadBundle(UInt(8.W), 1)))
        io.a := 0.U
      } }
    }
  }

  "Wrapped IO construction without parent reference" should "not fail for autoclonetype" in {
    class TestModule extends Module {
      def thunk[T](f: => T): T = f
      val works = thunk(IO(new Bundle {
        val x = Output(UInt(3.W))
      }))
    }
    elaborate { new TestModule }
  }

  "Wrapped IO construction with parent references" should "not fail for autoclonetype" in {
    class TestModule(blah: Int) extends Module {
      // Note that this currently fails only if f: =>T on Scala 2.11.12
      // This works successfully with 2.12.11
      def thunk[T](f: => T): T = f
      val broken = thunk(IO(new Bundle {
        val x = Output(UInt(blah.W))
      }))
    }
    elaborate { new TestModule(3) }
  }

  "Autoclonetype" should "support Bundles with if-blocks" in {
    class MyModule(n: Int) extends MultiIOModule {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
        if (n > 4) {
          println("Here we are!")
        }
      })
      io.out := io.in
    }
    elaborate(new MyModule(3))
  }

  behavior of "Compiler Plugin Autoclonetype"

  it should "NOT break code that extends chisel3.util Bundles if they use the plugin" in {
    class MyModule extends MultiIOModule {
      val io = IO(new InheritingBundle)
      io.deq <> io.enq
      io.count := 0.U
      io.error := true.B
    }
    elaborate(new MyModule)
  }

  it should "support Bundles with non-val parameters" in {
    class MyBundle(i: Int) extends Bundle {
      val foo = UInt(i.W)
    }
    elaborate { new MultiIOModule {
      val in = IO(Input(new MyBundle(8)))
      val out = IO(Output(new MyBundle(8)))
      out := in
    }}
  }

  it should "support type-parameterized Bundles" in {
    class MyBundle[T <: Data](gen: T) extends Bundle {
      val foo = gen
    }
    elaborate { new MultiIOModule {
      val in = IO(Input(new MyBundle(UInt(8.W))))
      val out = IO(Output(new MyBundle(UInt(8.W))))
      out := in
    }}
  }

  it should "support Bundles with non-val implicit parameters" in {
    class MyBundle(implicit i: Int) extends Bundle {
      val foo = UInt(i.W)
    }
    elaborate { new MultiIOModule {
      implicit val x = 8
      val in = IO(Input(new MyBundle))
      val out = IO(Output(new MyBundle))
      out := in
    }}
  }

  it should "support Bundles with multiple parameter lists" in {
    class MyBundle(i: Int)(j: Int, jj: Int)(k: UInt) extends Bundle {
      val foo = UInt((i + j + jj + k.getWidth).W)
    }
    elaborate {
      new MultiIOModule {
        val in = IO(Input(new MyBundle(8)(8, 8)(UInt(8.W))))
        val out = IO(Output(new MyBundle(8)(8, 8)(UInt(8.W))))
        out := in
      }
    }
  }

  it should "support Bundles that implement their own cloneType" in {
    class MyBundle(i: Int) extends Bundle {
      val foo = UInt(i.W)
    }
    elaborate { new MultiIOModule {
      val in = IO(Input(new MyBundle(8)))
      val out = IO(Output(new MyBundle(8)))
      out := in
    }}
  }

  it should "support Bundles that capture type parameters from their parent scope" in {
    class MyModule[T <: Data](gen: T) extends MultiIOModule {
      class MyBundle(n: Int) extends Bundle {
        val foo = Vec(n, gen)
      }
      val in = IO(Input(new MyBundle(4)))
      val out = IO(Output(new MyBundle(4)))
      out := in
    }
    elaborate(new MyModule(UInt(8.W)))
  }

  it should "work for higher-kinded types" in {
    class DataGen[T <: Data](gen: T) {
      def newType: T = gen.cloneType
    }
    class MyBundle[A <: Data, B <: DataGen[A]](gen: B) extends Bundle {
      val foo = gen.newType
    }
    class MyModule extends MultiIOModule {
      val io = IO(Output(new MyBundle[UInt, DataGen[UInt]](new DataGen(UInt(3.W)))))
      io.foo := 0.U
    }
    elaborate(new MyModule)
  }
}
