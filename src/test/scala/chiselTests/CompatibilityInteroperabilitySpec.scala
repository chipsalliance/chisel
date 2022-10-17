// SPDX-License-Identifier: Apache-2.0

package chiselTests

import scala.collection.immutable.ListMap

// Keep Chisel._ separate from chisel3._ below
object CompatibilityComponents {
  import Chisel._
  import Chisel3Components._

  class ChiselBundle extends Bundle {
    val a = UInt(width = 32)
    val b = UInt(width = 32).flip
  }
  class ChiselRecord extends Record {
    val elements = ListMap("a" -> UInt(width = 32), "b" -> UInt(width = 32).flip)
    override def cloneType: this.type = (new ChiselRecord).asInstanceOf[this.type]
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
  }

  class Chisel3Record extends Record {
    val elements = ListMap("a" -> Output(UInt(32.W)), "b" -> Input(UInt(32.W)))
    override def cloneType: this.type = (new Chisel3Record).asInstanceOf[this.type]
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

class CompatibilityInteroperabilitySpec extends ChiselFlatSpec {

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
        val children =
          Seq(Module(new PassthroughModule), Module(new PassthroughMultiIOModule), Module(new PassthroughRawModule))
        io.out := children.map(_.io.out).reduce(_ + _)
        children.foreach { child =>
          when(io.cond) {
            child.io.in := io.in
          }
        }
      }
    }
  }

  "Compatibility Modules" should "have Bool as their reset type" in {
    compile {
      import Chisel._
      class Intf extends Bundle {
        val in = Bool(INPUT)
        val en = Bool(INPUT)
        val out = Bool(OUTPUT)
      }
      class Child extends Module {
        val io = new Intf
        io.out := Mux(io.en, io.in, reset)
      }
      new Module {
        val io = new Intf
        val child = Module(new Child)
        io <> child.io
      }
    }
  }

  "Compatibility Modules" should "be instantiable inside chisel3 Modules" in {
    compile {
      object Compat {
        import Chisel._
        class Intf extends Bundle {
          val in = Input(UInt(8.W))
          val out = Output(UInt(8.W))
        }
        class OldMod extends Module {
          val io = IO(new Intf)
          io.out := Reg(next = io.in)
        }
      }
      import chisel3._
      import Compat._
      new Module {
        val io = IO(new Intf)
        io <> Module(new Module {
          val io = IO(new Intf)
          val inst = Module(new OldMod)
          io <> inst.io
        }).io
      }
    }
  }

  "A chisel3 Bundle that instantiates a Chisel Bundle" should "bulk connect correctly" in {
    compile {
      object Compat {
        import Chisel._
        class BiDir extends Bundle {
          val a = Input(UInt(8.W))
          val b = Output(UInt(8.W))
        }
        class Struct extends Bundle {
          val a = UInt(8.W)
        }
      }
      import chisel3._
      import Compat._
      class Bar extends Bundle {
        val bidir1 = new BiDir
        val bidir2 = Flipped(new BiDir)
        val struct1 = Output(new Struct)
        val struct2 = Input(new Struct)
      }
      // Check every connection both ways to see that chisel3 <>'s commutativity holds
      class Child extends RawModule {
        val deq = IO(new Bar)
        val enq = IO(Flipped(new Bar))
        enq <> deq
        deq <> enq
      }
      new RawModule {
        val deq = IO(new Bar)
        val enq = IO(Flipped(new Bar))
        // Also important to check connections to child ports
        val c1 = Module(new Child)
        val c2 = Module(new Child)
        c1.enq <> enq
        enq <> c1.enq
        c2.enq <> c1.deq
        c1.deq <> c2.enq
        deq <> c2.deq
        c2.deq <> deq
      }
    }
  }

  "A unidirectional but flipped Bundle" should "bulk connect in import chisel3._ code correctly" in {
    object Compat {
      import Chisel._
      class MyBundle(extraFlip: Boolean) extends Bundle {
        private def maybeFlip[T <: Data](t: T): T = if (extraFlip) t.flip else t
        val foo = maybeFlip(new Bundle {
          val bar = UInt(INPUT, width = 8)
        })
      }
    }
    import chisel3._
    import Compat._
    class Top(extraFlip: Boolean) extends RawModule {
      val port = IO(new MyBundle(extraFlip))
      val wire = Wire(new MyBundle(extraFlip))
      port <> DontCare
      wire <> DontCare
      port <> wire
      wire <> port
      port.foo <> wire.foo
      wire.foo <> port.foo
    }
    compile(new Top(true))
    compile(new Top(false))
  }

  "A unidirectional but flipped Bundle with something close to NotStrict compileOptions, but not exactly" should "bulk connect in import chisel3._ code correctly" in {
    object Compat {
      import Chisel.{defaultCompileOptions => _, _}
      // arbitrary thing to make this *not* exactly NotStrict
      implicit val defaultCompileOptions = new chisel3.ExplicitCompileOptions.CompileOptionsClass(
        connectFieldsMustMatch = false,
        declaredTypeMustBeUnbound = false,
        dontTryConnectionsSwapped = false,
        dontAssumeDirectionality = false,
        checkSynthesizable = false,
        explicitInvalidate = false,
        inferModuleReset = true // different from NotStrict, to ensure case class equivalence to NotStrict is false
      ) {
        override def emitStrictConnects = false
      }

      class MyBundle(extraFlip: Boolean) extends Bundle {
        private def maybeFlip[T <: Data](t: T): T = if (extraFlip) t.flip else t
        val foo = maybeFlip(new Bundle {
          val bar = UInt(INPUT, width = 8)
        })
      }
    }
    import chisel3._
    import Compat.{defaultCompileOptions => _, _}
    class Top(extraFlip: Boolean) extends RawModule {
      val port = IO(new MyBundle(extraFlip))
      val wire = Wire(new MyBundle(extraFlip))
      port <> DontCare
      wire <> DontCare
      port <> wire
      wire <> port
      port.foo <> wire.foo
      wire.foo <> port.foo
    }
    compile(new Top(true))
    compile(new Top(false))
  }

  "A undirectioned Chisel.Bundle used in a MixedVec " should "bulk connect in import chisel3._ code correctly" in {

    object UndirectionedBundleWithVagueCompileOptions {
      import chisel3.{Output, Bool, Vec}

      def bundle() = Output(Vec(3, Bool()))
    }

    object Compat {

      import Chisel._
      import chisel3.{WireInit, DontCare, RawModule}
      import chisel3.util.{MixedVec}
      import chisel3.experimental.hierarchy.{instantiable, public}

      class Node {
        def dangleGen(bundleGen: () => Data): Seq[(String, Data, Boolean)] = 
       {
           Seq(("in", WireInit(bundleGen(), DontCare), true),
           ("out", WireInit(bundleGen(), DontCare), false))
        }
      }

      /** [[AutoBundle]] will construct the [[Bundle]]s for a [[LazyModule]] in [[LazyModuleImpLike.instantiate]],
        *
        * @param elts is a sequence of data containing for each IO port  a tuple of (name, data, flipped), where
        *             name: IO name
        *             data: actual data for connection.
        *             flipped: flip or not in [[makeElements]]
        */
      final class AutoBundle(elts: (String, Data, Boolean)*) extends Record {
        // We need to preserve the order of elts, despite grouping by name to disambiguate things.
        val elements: ListMap[String, Data] = ListMap() ++ elts.zipWithIndex
          .map(makeElements)
          .groupBy(_._1)
          .values
          .flatMap {
            // If name is unique, it will return a Seq[index -> (name -> data)].
            case Seq((key, element, i)) => Seq(i -> (key -> element))
            // If name is not unique, name will append with j, and return `Seq[index -> (s"${name}_${j}" -> data)]`.
            case seq => seq.zipWithIndex.map { case ((key, element, i), j) => i -> (key + "_" + j -> element) }
          }
          .toList
          .sortBy(_._1)
          .map(_._2)
         require(elements.size == elts.size)

        // Trim final "(_[0-9]+)*$" in the name, flip data with flipped.
        private def makeElements(tuple: ((String, Data, Boolean), Int)) = {
          val ((key, data, flip), i) = tuple
          // Trim trailing _0_1_2 stuff so that when we append _# we don't create collisions.
          // Translate from Chisel2-style "default is Output" to explicit chisel3 directions
          val datax = data.cloneType match {
            case elt: Element if flip   => Input(elt)
            case elt: Element           => Output(elt)
            case agg: Aggregate if flip => Flipped(agg)
            case agg: Aggregate         => agg
          }

          (key, datax, i)
        }

        override def cloneType: this.type = new AutoBundle(elts: _*).asInstanceOf[this.type]
      }

      @instantiable
      sealed trait LazyModuleImpLike extends RawModule {

       def bundleGen: () => Data

       val node = new Node()
       val dangles = node.dangleGen(bundleGen)

       @public val auto = IO(new AutoBundle(dangles:_*))
      }

       class LazyModuleImp(val bundleGen: () => Data) extends chisel3.Module with LazyModuleImpLike

       @instantiable
      class MyModule() extends LazyModuleImp(bundleGen = () =>  UndirectionedBundleWithVagueCompileOptions.bundle()){
        @public val in = auto.elements("in")
        @public val out = auto.elements("out")
      }
      }

    object Chisel3 {
      import chisel3._
      import chisel3.experimental.hierarchy.{public, instantiable, Instance}

      @instantiable
      class MyModule
          extends Compat.LazyModuleImp(
            bundleGen = () => { UndirectionedBundleWithVagueCompileOptions.bundle() }
          ) {
                    @public val in = auto.elements("in")
        @public val out = auto.elements("out")
          }

      class Example extends Module {
        val oldMod = Module(new Compat.MyModule)
        val newMod = Module(new MyModule)

        oldMod.in <> DontCare
        newMod.in <> DontCare

        /*
        oldMod.inHead <> newMod.outHead
        newMod.inHead <> oldMod.outHead

        oldMod.autoInHead <> newMod.autoOutHead
        newMod.autoInHead <> oldMod.autoOutHead
*/
        val oldInst = Instance(oldMod.toDefinition)
        val newInst = Instance(newMod.toDefinition)
        oldInst.in <> DontCare
        newInst.in <> DontCare
      }
    }
    (new chisel3.stage.ChiselStage).emitVerilog(new Chisel3.Example, Array("--full-stacktrace"))
  }
}
