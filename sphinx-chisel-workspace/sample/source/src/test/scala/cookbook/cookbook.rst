-----------------------
src/test/scala/cookbook
-----------------------

.. toctree::


FSM.scala
---------
.. chisel:attr:: class DetectTwoOnes extends Module


.. chisel:attr:: class DetectTwoOnesTester extends CookbookTester(10)


.. chisel:attr:: class FSMSpec extends CookbookSpec


UInt2Bundle.scala
-----------------
.. chisel:attr:: class UInt2Bundle extends CookbookTester(1)


.. chisel:attr:: class UInt2BundleSpec extends CookbookSpec


VecOfBool2UInt.scala
--------------------
.. chisel:attr:: class VecOfBool2UInt extends CookbookTester(1)


.. chisel:attr:: class VecOfBool2UIntSpec  extends CookbookSpec


CookbookSpec.scala
------------------
.. chisel:attr:: abstract class CookbookTester(length: Int) extends BasicTester

	Tester for concise cookbook tests	
	Provides a length of test after which the test will pass
  

.. chisel:attr:: abstract class CookbookSpec extends ChiselFlatSpec  


Bundle2UInt.scala
-----------------
.. chisel:attr:: class Bundle2UInt extends CookbookTester(1)


.. chisel:attr:: class Bundle2UIntSpec  extends CookbookSpec


RegOfVec.scala
--------------
.. chisel:attr:: class RegOfVec extends CookbookTester(2)


.. chisel:attr:: class RegOfVecSpec  extends CookbookSpec


UInt2VecOfBool.scala
--------------------
.. chisel:attr:: class UInt2VecOfBool extends CookbookTester(1)


.. chisel:attr:: class UInt2VecOfBoolSpec extends CookbookSpec


