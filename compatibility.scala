package Chisel

import chisel3.{ iotesters => ciot }

/**
  * Provide "Chisel" interface to specific chisel3 internals.
  */
package object iotesters {
  type ChiselFlatSpec = ciot.ChiselFlatSpec
  type ChiselPropSpec = ciot.ChiselPropSpec
  type PeekPokeTester[+T <: Module] = ciot.PeekPokeTester[T]
  type Backend = ciot.Backend
  type HWIOTester = ciot.HWIOTester
  type SteppedHWIOTester = ciot.SteppedHWIOTester
  type OrderedDecoupledHWIOTester = ciot.OrderedDecoupledHWIOTester

  object chiselMainTest {
    def apply[T <: Module](args: Array[String], dutGen: () => T)(testerGen: T => ciot.PeekPokeTester[T]) = {
      ciot.chiselMain(args, dutGen, testerGen)
    }
  }
  
  /**
    * Runs the ClassicTester and returns a Boolean indicating test success or failure
    * @@backendType determines whether the ClassicTester uses verilator or the firrtl interpreter to simulate the circuit
    * Will do intermediate compliation steps to setup the backend specified, including cpp compilation for the verilator backend and firrtl IR compilation for the firrlt backend
    */
  object runPeekPokeTester {
    def apply[T <: Module](dutGen: () => T, backendType: String = "firrtl")(testerGen: (T, Option[ciot.Backend]) => ciot.PeekPokeTester[T]): Boolean = {
      ciot.runPeekPokeTester(dutGen, backendType)(testerGen)
    }
  }
  
  /**
    * Runs the ClassicTester using the verilator backend without doing Verilator compilation and returns a Boolean indicating success or failure
    * Requires the caller to supply path the already compile Verilator binary
    */
  object runPeekPokeTesterWithVerilatorBinary {
    def apply[T <: Module](dutGen: () => T, verilatorBinaryFilePath: String)(testerGen: (T, Option[ciot.Backend]) => ciot.PeekPokeTester[T]): Boolean = {
      ciot.runPeekPokeTesterWithVerilatorBinary(dutGen, verilatorBinaryFilePath)(testerGen)
    }
  }
}
