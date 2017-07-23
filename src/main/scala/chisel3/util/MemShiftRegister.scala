
package chisel3.util

import chisel3._

/** This is functionally identical to a ShiftRegister but uses a Memory for storage.
  * If the value of n <= 2 it will revert to the ShiftRegister implementation.
  */
object MemShiftRegister {

  /** Returns the n-cycle delayed version of the input signal with reset initialization.
    *
    * @param in input to delay
    * @param n number of cycles to delay
    * @param en enable the shift
    */
  def apply[ T <: Data ]( in : T, n : Int, en : Bool = true.B ) : T = {
    val memSR = Module( new MemShiftRegister( in, n ) )
    memSR.io.en := en
    memSR.io.in := in
    memSR.io.out
  }

  /** Returns the n-cycle delayed version of the input signal with reset initialization.
    *
    * @param in input to delay
    * @param n number of cycles to delay
    * @param resetData reset value for each register in the shift
    * @param en enable the shift
    */
  def apply[ T <: Data ]( in : T, n : Int, resetData : T, en : Bool ) : T = {
    if ( n <= 2 )
      ShiftRegister( in, n, resetData, en )
    else {
      val memSR = Module( new MemShiftRegister( in, n ) )
      memSR.io.en := en
      memSR.io.in := in
      val initDone = RegInit( false.B )
      initDone := initDone | memSR.io.cntrWrap
      val out = Wire( resetData.cloneType )
      out := resetData
      when ( initDone ) {
        out := memSR.io.out
      }
      out
    }
  }

}

/** Do not use this class directly, instead use [[chisel3.util.MemShiftRegister$]] Factory method
  */
class MemShiftRegister[ T <: Data ]( genType : T, n : Int ) extends Module {
  val io = IO(new Bundle {
    val in = Input( genType.cloneType )
    val en = Input( Bool() )
    val cntrWrap = Output( Bool() )
    val out = Output( genType.cloneType )
  })

  if ( n <= 2 )
    io.out := ShiftRegister( io.in, n, io.en )
  else {
    val myMem = SyncReadMem( n, genType.cloneType )

    val cntr = Counter( io.en, n )
    val readAddr = Wire( UInt( cntr._1.getWidth.W + 1.W ) )

    readAddr := cntr._1 + 1.U
    when ( cntr._1 === ( n - 1 ).U ) {
      readAddr := 0.U
    }

    when ( io.en ) {
      myMem.write( cntr._1, io.in )
    }
    io.out := myMem.read( readAddr, io.en )
    io.cntrWrap := cntr._2
  }
}
