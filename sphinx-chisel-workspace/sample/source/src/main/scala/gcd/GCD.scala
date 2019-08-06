// See README.md for license details.

package gcd

import chisel3._

class MyBundle extends Bundle {
	val value1        = Input(UInt(16.W))
	val value2        = Input(UInt(16.W))
	val loadingValues = Input(Bool())
	val outputGCD     = Output(UInt(16.W))
	val outputValid   = Output(Bool())
        val otherBundle   = Output(new MyBundleTwo)
}

class MyBundleTwo extends Bundle {
	val value3        = Input(UInt(16.W))
	val value4        = Input(UInt(16.W))
	val outputTwo     = Output(UInt(16.W))
}

/**
  * Compute GCD using subtraction method.
  * Subtracts the smaller from the larger until register y is zero.
  * value in register x is then the GCD
  * 
  * @io io A [[Bundle]] containing the values used to compute the GCD.
  * @ioattr io.value1 The first input to the GCD algorithm.
  * @ioattr io.value2 The second input to the GCD algorithm.
  * @ioattr io.loadingValues Signals whether or not the values have been loaded.  
  * @ioattr io.outputGCD : chisel3.core.UInt The output of the GCD algorithm.
  * @ioattr io.outputValid Signals whether the output has now become valid.  
  * @ioattr io.otherBundle A custom bundle added for testing.
  * @ioattr io.otherBundle.value3 An input field of otherBundle created for tests.
  * @ioattr io.otherBundle.value4 Another input field of otherBundle created for tests.
  * @ioattr io.otherBundle.outputTwo The output of otherBundle.
  */
class GCD extends Module {
  val io = IO(new MyBundle)

  val x  = Reg(UInt())
  val y  = Reg(UInt())

  when(x > y) { x := x - y }
    .otherwise { y := y - x }

  when(io.loadingValues) {
    x := io.value1
    y := io.value2
  }

  io.outputGCD := x
  io.outputValid := y === 0.U
}
