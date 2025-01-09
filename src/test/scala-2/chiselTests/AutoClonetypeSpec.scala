// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.QueueIO
import circt.stage.ChiselStage.emitCHIRRTL

import scala.collection.immutable.ListMap

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

  val io = IO(new Bundle {})

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

class RecordAutoCloneType[T <: Data](gen: T) extends Record {
  lazy val elements = ListMap("value" -> gen)
  // This is a weird thing to do, but as only Bundles have these methods, it should be legal
  protected def _elementsImpl: Iterable[(String, Any)] = elements
  protected def _usingPlugin = false
}

// Records that don't mixin AutoCloneType should still be able to implement the related methods
// NOTE: This is a very weird thing to do, don't do it.
class RecordWithVerbotenMethods(w: Int) extends Record {
  lazy val elements = ListMap("value" -> UInt(w.W))
  // Verboten methods
  protected def _usingPlugin = false

  protected def _elementsImpl: Iterable[(String, Any)] = Nil
}

class AutoClonetypeSpec extends ChiselFlatSpec with Utils {

  "Bundles with Scala args" should "not need clonetype" in {
    emitCHIRRTL {
      new Module {
        val io = IO(new Bundle {})

        val myWire = Wire(new BundleWithIntArg(8))
        assert(myWire.i == 8)
      }
    }
  }

  "Bundles with Scala implicit args" should "not need clonetype" in {
    emitCHIRRTL {
      new Module {
        val io = IO(new Bundle {})

        implicit val implicitInt: Int = 4
        val myWire = Wire(new BundleWithImplicit())

        assert(myWire.ii == 4)
      }
    }
  }

  "Bundles with Scala explicit and impicit args" should "not need clonetype" in {
    emitCHIRRTL {
      new Module {
        val io = IO(new Bundle {})

        implicit val implicitInt: Int = 4
        val myWire = Wire(new BundleWithArgAndImplicit(8))

        assert(myWire.i == 8)
        assert(myWire.ii == 4)
      }
    }
  }

  "Subtyped Bundles" should "not need clonetype" in {
    emitCHIRRTL {
      new Module {
        val io = IO(new Bundle {})

        val myWire = Wire(new SubBundle(8, 4))

        assert(myWire.i == 8)
        assert(myWire.i2 == 4)
      }
    }
    emitCHIRRTL {
      new Module {
        val io = IO(new Bundle {})

        val myWire = Wire(new SubBundleVal(8, 4))

        assert(myWire.i == 8)
        assert(myWire.i2 == 4)
      }
    }
  }

  "Autoclonetype" should "work outside of a builder context" in {
    new BundleWithIntArg(8).cloneType
  }

  "Subtyped Bundles that don't clone well" should "be now be supported!" in {
    emitCHIRRTL {
      new Module {
        val io = IO(new Bundle {})
        val myWire = Wire(new SubBundleInvalid(8, 4))
      }
    }
  }

  "Inner bundles with Scala args" should "not need clonetype" in {
    emitCHIRRTL { new ModuleWithInner }
  }

  "Bundles with arguments as fields" should "not need clonetype" in {
    emitCHIRRTL {
      new Module {
        val io = IO(Output(new BundleWithArgumentField(UInt(8.W), UInt(8.W))))
        io.x := 1.U
        io.y := 1.U
      }
    }
  }

  it should "also work when giving directions to the fields" in {
    emitCHIRRTL {
      new Module {
        val io = IO(new BundleWithArgumentField(Input(UInt(8.W)), Output(UInt(8.W))))
        io.y := io.x
      }
    }
  }

  "Bundles inside companion objects" should "not need clonetype" in {
    emitCHIRRTL {
      new Module {
        val io = IO(Output(new CompanionObjectWithBundle.Inner))
        io.data := 1.U
      }
    }
  }

  "Parameterized bundles inside companion objects" should "not need clonetype" in {
    emitCHIRRTL {
      new Module {
        val io = IO(Output(new CompanionObjectWithBundle.ParameterizedInner(8)))
        io.data := 1.U
      }
    }
  }

  "Nested directioned anonymous Bundles" should "not need clonetype" in {
    emitCHIRRTL {
      new Module {
        val io = IO(new NestedAnonymousBundle)
        val a = WireDefault(io)
        io.a.a := 1.U
      }
    }
  }

  "3.0 null compatibility" should "not need clonetype" in {
    emitCHIRRTL {
      new Module {
        class InnerClassThing {
          def createBundle: Bundle = new Bundle {
            val a = Output(UInt(8.W))
          }
        }
        val io = IO((new InnerClassThing).createBundle)
        val a = WireDefault(io)
      }
    }
  }

  "Aliased fields" should "be caught" in {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      emitCHIRRTL {
        new Module {
          val bundleFieldType = UInt(8.W)
          val io = IO(Output(new Bundle {
            val a = bundleFieldType
          }))
          io.a := 0.U
        }
      }
    }
  }

  "Aliased fields from inadequate autoclonetype" should "be caught" in {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      class BadBundle(val typeTuple: (Data, Int)) extends Bundle {
        val a = typeTuple._1
      }

      emitCHIRRTL {
        new Module {
          // This needs to be constructed before the call to Output, otherwise it won't be cloned
          // thanks to lazy cloning
          val gen = new BadBundle(UInt(8.W), 1)
          val io = IO(Output(gen))
          io.a := 0.U
        }
      }
    }
  }

  "Wrapped IO construction without parent reference" should "not fail for autoclonetype" in {
    class TestModule extends Module {
      def thunk[T](f: => T): T = f
      val works = thunk(IO(new Bundle {
        val x = Output(UInt(3.W))
      }))
    }
    emitCHIRRTL { new TestModule }
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
    emitCHIRRTL { new TestModule(3) }
  }

  "Autoclonetype" should "support Bundles with if-blocks" in {
    class MyModule(n: Int) extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
        if (n > 4) {
          println("Here we are!")
        }
      })
      io.out := io.in
    }
    emitCHIRRTL(new MyModule(3))
  }

  behavior.of("Compiler Plugin Autoclonetype")

  it should "NOT break code that extends chisel3.util Bundles if they use the plugin" in {
    class MyModule extends Module {
      val io = IO(new InheritingBundle)
      io.deq <> io.enq
      io.count := 0.U
      io.error := true.B
    }
    emitCHIRRTL(new MyModule)
  }

  it should "support Bundles with non-val parameters" in {
    class MyBundle(i: Int) extends Bundle {
      val foo = UInt(i.W)
    }
    emitCHIRRTL {
      new Module {
        val in = IO(Input(new MyBundle(8)))
        val out = IO(Output(new MyBundle(8)))
        out := in
      }
    }
  }

  it should "support type-parameterized Bundles" in {
    class MyBundle[T <: Data](gen: T) extends Bundle {
      val foo = gen
    }
    emitCHIRRTL {
      new Module {
        val in = IO(Input(new MyBundle(UInt(8.W))))
        val out = IO(Output(new MyBundle(UInt(8.W))))
        out := in
      }
    }
  }

  it should "support Bundles with non-val implicit parameters" in {
    class MyBundle(implicit i: Int) extends Bundle {
      val foo = UInt(i.W)
    }
    emitCHIRRTL {
      new Module {
        implicit val x = 8
        val in = IO(Input(new MyBundle))
        val out = IO(Output(new MyBundle))
        out := in
      }
    }
  }

  it should "support Bundles with multiple parameter lists" in {
    class MyBundle(i: Int)(j: Int, jj: Int)(k: UInt) extends Bundle {
      val foo = UInt((i + j + jj + k.getWidth).W)
    }
    emitCHIRRTL {
      new Module {
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
    emitCHIRRTL {
      new Module {
        val in = IO(Input(new MyBundle(8)))
        val out = IO(Output(new MyBundle(8)))
        out := in
      }
    }
  }

  it should "support Bundles that capture type parameters from their parent scope" in {
    class MyModule[T <: Data](gen: T) extends Module {
      class MyBundle(n: Int) extends Bundle {
        val foo = Vec(n, gen)
      }
      val in = IO(Input(new MyBundle(4)))
      val out = IO(Output(new MyBundle(4)))
      out := in
    }
    emitCHIRRTL(new MyModule(UInt(8.W)))
  }

  it should "work for higher-kinded types" in {
    class DataGen[T <: Data](gen: T) {
      def newType: T = gen.cloneType
    }
    class MyBundle[A <: Data, B <: DataGen[A]](gen: B) extends Bundle {
      val foo = gen.newType
    }
    class MyModule extends Module {
      val io = IO(Output(new MyBundle[UInt, DataGen[UInt]](new DataGen(UInt(3.W)))))
      io.foo := 0.U
    }
    emitCHIRRTL(new MyModule)
  }

  it should "support Bundles with vararg arguments" in {
    // Without the fix, this doesn't even compile
    // Extra parameter lists to make this a complex test case
    class VarArgsBundle(x: Int)(y: Int, widths: Int*) extends Bundle {
      def mkField(idx: Int): Option[UInt] =
        (x +: y +: widths).lift(idx).map(w => UInt(w.W))
      val foo = mkField(0)
      val bar = mkField(1)
      val fizz = mkField(2)
      val buzz = mkField(3)
    }
    class MyModule extends Module {
      val in = IO(Input(new VarArgsBundle(1)(2, 3, 4)))
      val out = IO(Output(new VarArgsBundle(1)(2, 3, 4)))
      out := in
    }
    emitCHIRRTL(new MyModule)
  }

  it should "support Records that mixin AutoCloneType" in {
    class MyModule extends Module {
      val gen = new RecordAutoCloneType(UInt(8.W))
      val in = IO(Input(gen))
      val out = IO(Output(gen))
      out := in
    }
    emitCHIRRTL(new MyModule)
  }

  it should "support Records that don't mixin AutoCloneType and use forbidden methods" in {
    class MyModule extends Module {
      val gen = new RecordWithVerbotenMethods(8)
      val in = IO(Input(gen))
      val out = IO(Output(gen))
      out := in
    }
    emitCHIRRTL(new MyModule)
  }

  it should "compile with package private default bundle constructors" in {
    class PrivateDefaultConsBundle private[chiselTests] (w: Int) extends Bundle {
      val x = UInt(w.W)
    }
    object PrivateDefaultConsBundle {
      def apply(w: Int): PrivateDefaultConsBundle = new PrivateDefaultConsBundle(w)
    }
    class MyModule extends Module {
      val in = IO(Input(PrivateDefaultConsBundle(8)))
      val out = IO(Output(PrivateDefaultConsBundle(8)))
      out := in
    }
    emitCHIRRTL(new MyModule)
  }
}
