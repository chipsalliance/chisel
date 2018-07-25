package chisel3.libs.diagnostic

import chisel3._
import chisel3.core.{ChiselAnnotation, dontTouch}
import chisel3.experimental.MultiIOModule
import chisel3.libs.aspect.ModuleAspect
import firrtl.ir.{Input => _, Module => _, Output => _, _}

/**
  * Generate annotations to inject hardware to track a histogram of a signal's values
  */
object Histogram {

  /**
    *
    * @param name Name of the hardware histogram instance
    * @param root Location where the breakpoint will live
    * @param signal Signal to histogram values
    * @param maxCount Max number of cycles to track a given signal value
    * @tparam T Type of the root hardware
    * @tparam S Type of the signal
    * @return Chisel annotations
    */
  def apply[T<: MultiIOModule, S<:Bits](name: String, root: T, signal: S, maxCount: Int): Seq[ChiselAnnotation] = {

    ModuleAspect(name, root, () => new Histogram(chiselTypeOf(signal), maxCount), (encl: T, hist: Histogram[S]) => {
      Map(
        encl.clock -> hist.clock,
        encl.reset -> hist.reset,
        signal -> hist.in
      )
    })
  }
}

class Histogram[T<:Bits](signalType: => T, maxCount: Int) extends MultiIOModule {
  val in = IO(Input(signalType))
  val out = IO(Output(signalType))
  val histMem = Mem(math.pow(2, in.getWidth).toInt, chiselTypeOf(maxCount.U))
  val readPort = histMem.read(in.asUInt())
  histMem.write(in.asUInt(), (readPort + 1.U).min(maxCount.U))
  out := readPort
  dontTouch(out)
}
