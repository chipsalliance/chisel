--------------------------------------
src/test/scala/chiselTests/util/random
--------------------------------------

.. toctree::


PRNGSpec.scala
--------------
.. chisel:attr:: class CyclePRNG(width: Int, seed: Option[BigInt], step: Int, updateSeed: Boolean) extends PRNG(width, seed, step, updateSeed)


.. chisel:attr:: class PRNGStepTest extends BasicTester


.. chisel:attr:: class PRNGUpdateSeedTest(updateSeed: Boolean, seed: BigInt, expected: BigInt) extends BasicTester


.. chisel:attr:: class PRNGSpec extends ChiselFlatSpec


LFSRSpec.scala
--------------
.. chisel:attr:: class FooLFSR(val reduction: LFSRReduce, seed: Option[BigInt]) extends PRNG(4, seed) with LFSR


.. chisel:attr:: class LFSRResetTester(gen: => LFSR, lockUpValue: BigInt) extends BasicTester

	This tests that after reset an LFSR is not locked up. This manually sets the seed of the LFSR at run-time to the	value that would cause it to lock up. It then asserts reset. The next cycle it checks that the value is NOT the
	locked up value.
	
	:param gen: an LFSR to test
	
	:param lockUpValue: the value that would lock up the LFSR
	  

.. chisel:attr:: class LFSRSpec extends ChiselFlatSpec


