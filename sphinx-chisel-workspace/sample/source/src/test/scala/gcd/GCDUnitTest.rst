-----------------
GCDUnitTest.scala
-----------------
.. chisel:attr:: class GCDUnitTester(c: GCD) extends PeekPokeTester(c)


	.. chisel:attr:: def computeGcd(a: Int, b: Int): (Int, Int) =

	
			compute the gcd and the number of steps it should take to do it
		
		
		:param a: positive integer
		
		:param b: positive integer
		:return: the GCD of a and b
		    


.. chisel:attr:: class GCDTester extends ChiselFlatSpec

		This is a trivial example of how to run this Specification
	From within sbt use:
	
	.. code-block:: scala 

		 testOnly gcd.GCDTester
	
	From a terminal shell use:
	
	.. code-block:: scala 

		 sbt 'testOnly gcd.GCDTester'
	
  

	.. chisel:attr:: "running with --generate-vcd-output on" should "create a vcd file from your test" in

	
			By default verilator backend produces vcd file, and firrtl and treadle backends do not.
		Following examples show you how to turn on vcd for firrtl and treadle and how to turn it off for verilator
	    


