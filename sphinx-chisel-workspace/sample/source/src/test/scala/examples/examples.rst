-----------------------
src/test/scala/examples
-----------------------

.. toctree::


SimpleVendingMachine.scala
--------------------------
.. chisel:attr:: class SimpleVendingMachineIO extends Bundle


.. chisel:attr:: abstract class SimpleVendingMachine extends Module


.. chisel:attr:: class FSMVendingMachine extends SimpleVendingMachine


.. chisel:attr:: class VerilogVendingMachine extends BlackBox


.. chisel:attr:: class VerilogVendingMachineWrapper extends SimpleVendingMachine


.. chisel:attr:: class SimpleVendingMachineTester(mod: => SimpleVendingMachine) extends BasicTester


.. chisel:attr:: class SimpleVendingMachineSpec extends ChiselFlatSpec


VendingMachineGenerator.scala
-----------------------------
.. chisel:attr:: class VendingMachineIO(val legalCoins: Seq[Coin]) extends Bundle


.. chisel:attr:: abstract class ParameterizedVendingMachine(legalCoins: Seq[Coin], val sodaCost: Int) extends Module


.. chisel:attr:: class VendingMachineGenerator(legalCoins: Seq[Coin], sodaCost: Int) extends ParameterizedVendingMachine(legalCoins, sodaCost)


.. chisel:attr:: class ParameterizedVendingMachineTester(mod: => ParameterizedVendingMachine, testLength: Int) extends BasicTester


.. chisel:attr:: class VendingMachineGeneratorSpec extends ChiselFlatSpec


ImplicitStateVendingMachine.scala
---------------------------------
.. chisel:attr:: class ImplicitStateVendingMachine extends SimpleVendingMachine


.. chisel:attr:: class ImplicitStateVendingMachineSpec extends ChiselFlatSpec


VendingMachineUtils.scala
-------------------------
.. chisel:attr:: object VendingMachineUtils


