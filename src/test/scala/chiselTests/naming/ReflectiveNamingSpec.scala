// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chiselTests.{ChiselFlatSpec, Utils}

class ReflectiveNamingSpec extends ChiselFlatSpec with Utils {

  behavior.of("Reflective naming")

  private def emitChirrtl(gen: => RawModule): String = {
    // Annoyingly need to emit files to use CLI
    val targetDir = createTestDirectory(this.getClass.getSimpleName).toString
    val args = Array("--warn:reflective-naming", "-td", targetDir)
    (new chisel3.stage.ChiselStage).emitChirrtl(gen, args)
  }

  it should "NOT warn when no names are changed" in {
    class Example extends Module {
      val foo, bar = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))

      val sum = foo +& bar
      out := sum
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("node sum = add(foo, bar)")
  }

  it should "warn when changing the name of a node" in {
    class Example extends Module {
      val foo, bar = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))

      val sum = foo +& bar
      val fuzz = sum
      out := sum
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should include("'sum' is renamed by reflection to 'fuzz'")
    chirrtl should include("node fuzz = add(foo, bar)")
  }

  // This also checks correct prefix reversing
  it should "warn when changing the name of a node with a prefix in the name" in {
    class Example extends Module {
      val foo, bar = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))

      // This is sketch, don't do this
      var fuzz = 0.U
      out := {
        val sum = {
          val node = foo +& bar
          fuzz = node
          node +% 0.U
        }
        sum
      }
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should include("'out_sum_node' is renamed by reflection to 'fuzz'")
    chirrtl should include("node fuzz = add(foo, bar)")
  }

  it should "warn when changing the name of a Module instance" in {
    import chisel3.util._
    class Example extends Module {
      val enq = IO(Flipped(Decoupled(UInt(8.W))))
      val deq = IO(Decoupled(UInt(8.W)))

      val q = Module(new Queue(UInt(8.W), 4))
      q.io.enq <> enq
      deq <> q.io.deq

      val fuzz = q
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should include("'q' is renamed by reflection to 'fuzz'")
    chirrtl should include("inst fuzz of Queue")
  }

  it should "warn when changing the name of an Instance" in {
    import chisel3.experimental.hierarchy.{Definition, Instance}
    import chiselTests.experimental.hierarchy.Examples.AddOne
    class Example extends Module {
      val defn = Definition(new AddOne)
      val inst = Instance(defn)
      val fuzz = inst
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should include("'inst' is renamed by reflection to 'fuzz'")
    chirrtl should include("inst fuzz of AddOne")
  }

  it should "warn when changing the name of a Mem" in {
    class Example extends Module {
      val mem = SyncReadMem(8, UInt(8.W))

      val fuzz = mem
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should include("'mem' is renamed by reflection to 'fuzz'")
    chirrtl should include("smem fuzz")
  }

  it should "NOT warn when changing the name of a verification statement" in {
    class Example extends Module {
      val in = IO(Input(UInt(8.W)))
      val z = chisel3.assert(in =/= 123.U)
      val fuzz = z
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should equal("")
    // But the name is actually changed
    (chirrtl should include).regex("assert.*: fuzz")
  }

  it should "NOT warn when \"naming\" a literal" in {
    class Example extends Module {
      val out = IO(Output(UInt(8.W)))

      val sum = 0.U
      val fuzz = sum
      out := sum
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("out <= UInt")
  }

  it should "NOT warn when \"naming\" a field of an Aggregate" in {
    class Example extends Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      val in = io.in
      val out = io.out
      out := in
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("io.out <= io.in")
  }

  it should "NOT warn when \"naming\" unbound Data" in {
    class Example extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      val z = UInt(8.W)
      val a = z
      out := in
    }
    val (log, chirrtl) = grabLog(emitChirrtl(new Example))
    log should equal("")
    chirrtl should include("out <= in")
  }
}
