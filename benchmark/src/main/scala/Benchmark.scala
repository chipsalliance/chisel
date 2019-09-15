package bench

import chisel3._
import scala.io.Source
import java.io.File

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations._

class NestedBundle(val w: Int) extends Bundle {
  class BundleA extends Bundle {
    class BundleB extends Bundle {
      class BundleC extends Bundle {
        class BundleD extends Bundle {
          val x = UInt(8.W)
        }
        val x = new BundleD
      }
      val x = new BundleC
    }
    val x = new BundleB
  }
  val x = new BundleA
}

class MyModule[T <: Data](gen: => T) extends Module {
  val io = IO(new Bundle {
    val in = Input(gen)
    val out = Output(gen)
  })
  io.out := io.in
}

@State(Scope.Thread)
//@OutputTimeUnit(TimeUnit.SECONDS)
//@Measurement(iterations = 10)
//@BenchmarkMode(Array(Mode.AverageTime))
@Fork(1)
//@Threads(1)
class BenchmarkCloneType {

  //var nestedBundle: NestedBundle = _

  //@Setup
  //def buildNestedBundle: Unit = nestedBundle = new NestedBundle(8)

  //@Benchmark
  //def nestedBundleBuild: Unit = new NestedBundle(8)

  //@Benchmark
  //def nestedBundleCloneType: Unit = nestedBundle.cloneType

  @Benchmark
  def baseModule: Unit = chisel3.Driver.elaborate(() => new MyModule(UInt(8.W)))

  @Benchmark
  def nestedBundle: Unit = chisel3.Driver.elaborate(() => new MyModule(new NestedBundle(8)))
}
