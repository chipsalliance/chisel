
package chisel3.util

import chisel3._

object MemShiftRegister {

  def apply[ T <: Data ]( in : T, n : Int, en : Bool = true.B ) : T = {
    val memSR = Module( new MemShiftRegister( in, n ) )
    memSR.io.en := en
    memSR.io.in := in
    memSR.io.out
  }

  def apply[ T <: Data ]( in : T, n : Int, resetData : T, en : Bool ) : T = {
    if ( n <= 2 )
      ShiftRegister( in, n, resetData, en )
    else {
      val memOut = apply( in, n, en )
      val initCntr = Counter( en, n )
      val initDone = RegInit( false.B )
      initDone := initDone | initCntr._2
      val out = Wire( resetData.cloneType )
      out := resetData
      when ( initDone ) {
        out := memOut
      }
      out
    }
  }

}

class MemShiftRegister[ T <: Data ]( genType : T, n : Int ) extends Module {
  val io = IO(new Bundle {
    val in = Input( genType.cloneType )
    val en = Input( Bool() )
    val out = Output( genType.cloneType )
  })

  if ( n <= 2 )
    io.out := ShiftRegister( io.in, n, io.en )
  else {
    val myMem = SyncReadMem( n - 1, genType.cloneType )

    // put a register at the front and back
    val regTop = RegEnable( io.in, io.en )
    val cntr = Counter( io.en, n - 1 )
    val readAddr = Wire( UInt( cntr._1.getWidth.W + 1.W ) )

    readAddr := cntr._1 + 1.U
    when ( cntr._1 === ( n - 2 ).U ) {
      readAddr := 0.U
    }

    when ( io.en ) {
      myMem.write( cntr._1, regTop )
    }
    io.out := myMem.read( readAddr, io.en )
  }
}
