---------
GCD.scala
---------
.. chisel:attr:: class MyBundle extends Bundle


.. chisel:attr:: class MyBundleTwo extends Bundle


.. chisel:attr:: class GCD extends Module

		Compute GCD using subtraction method.
	Subtracts the smaller from the larger until register y is zero.
	value in register x is then the GCD
	
	:IO: io A :chisel:reref:`Bundle`  containing the values used to compute the GCD.
	:IO attr: io.value1 :chisel:reref:`chisel3.core.UInt` - The first input to the GCD algorithm.
	:IO attr: io.value2 :chisel:reref:`chisel3.core.UInt` - The second input to the GCD algorithm.
	:IO attr: io.loadingValues :chisel:reref:`chisel3.core.Bool` - Signals whether or not the values have been loaded.
	:IO attr: io.outputGCD :chisel:reref:`chisel3.core.UInt` - The output of the GCD algorithm.
	:IO attr: io.outputValid :chisel:reref:`chisel3.core.Bool` - Signals whether the output has now become valid.
	:IO attr: io.otherBundle :chisel:reref:`gcd.MyBundleTwo` - A custom bundle added for testing.
	:IO attr: io.otherBundle.value3 :chisel:reref:`chisel3.core.UInt` - An input field of otherBundle created for tests.
	:IO attr: io.otherBundle.value4 :chisel:reref:`chisel3.core.UInt` - Another input field of otherBundle created for tests.
	:IO attr: io.otherBundle.outputTwo :chisel:reref:`chisel3.core.UInt` - The output of otherBundle.
  

