----------------------------------
src/main/scala/chisel3/util/random
----------------------------------

.. toctree::


LFSR.scala
----------
.. chisel:attr:: sealed trait LFSRReduce extends ((Bool, Bool) => Bool)

	A reduction operation for an LFSR.	@see :chisel:reref:`XOR`
	@see :chisel:reref:`XNOR`
  

.. chisel:attr:: object XOR extends LFSRReduce

	XOR (exclusive or) reduction operation 


.. chisel:attr:: object XNOR extends LFSRReduce

	Not XOR (exclusive or) reduction operation 


.. chisel:attr:: trait LFSR extends PRNG

	Trait that defines a Linear Feedback Shift Register (LFSR).	
	$seedExplanation
	@see :chisel:reref:`FibonacciLFSR`
	@see :chisel:reref:`GaloisLFSR`
	@see :chisel:reref:`https://en.wikipedia.org/wiki/Linear-feedback_shift_register`
	
	
@define paramWidth 	:param width: the width of the LFSR
	
@define paramTaps 	:param taps: a set of tap points to use when constructing the LFSR
	
@define paramSeed 	:param seed: an initial value for internal LFSR state. If :chisel:reref:`scala.None None` , then the LFSR
		state LSB will be set to a known safe value on reset (to prevent lock up).
	
@define paramReduction 	:param reduction: the reduction operation (either :chisel:reref:`chisel3.util.random.XOR XOR`  or
		:chisel:reref:`chisel3.util.random.XNOR XNOR` )
	
@define paramStep 	:param step: the number of state updates per cycle
	
@define paramUpdateSeed 	:param updateSeed: if true, when loading the seed the state will be updated as if the seed
		were the current state, if false, the state will be set to the seed
		@define seedExplanation If the user specifies a seed, then a compile-time check is added that they are not
		initializing the LFSR to a state which will cause it to lock up. If the user does not set a seed, then the least
		significant bit of the state will be set or reset based on the choice of reduction operator.
	  

	.. chisel:attr:: def reduction: LFSRReduce

	
		The binary reduction operation used by this LFSR, either :chisel:reref:`chisel3.util.random.XOR XOR`  or	:chisel:reref:`chisel3.util.random.XNOR XNOR` . This has the effect of mandating what seed is invalid.
	    


.. chisel:attr:: object LFSR

	Utilities related to psuedorandom number generation using Linear Feedback Shift Registers (LFSRs).	
	For example, to generate a pseudorandom 16-bit :chisel:reref:`UInt`  that changes every cycle, you can use:
	
	.. code-block:: scala 

		 val pseudoRandomNumber = LFSR(16)
	
  

	.. chisel:attr:: def apply(width: Int, increment: Bool = true.B, seed: Option[BigInt] = Some(1)): UInt = FibonacciLFSR.maxPeriod(width, increment, seed, XOR)

	
		Return a pseudorandom :chisel:reref:`UInt`  generated using a :chisel:reref:`FibonacciLFSR` . If you require a Galois LFSR, use	:chisel:reref:`GaloisLFSR$.maxPeriod GaloisLFSR.maxPeriod` .
		
		:param width: the width of the LFSR
		
		:param increment: when asserted, the LFSR will increment
		
		:param seed: an initial seed (this cannot be zero)
		:return: a :chisel:reref:`UInt`  that is the output of a maximal period LFSR of the requested width
		    


	.. chisel:attr:: private [random] def badWidth(width: Int): Nothing

	
		Utility used to report an unknown tap width 
	


	.. chisel:attr:: private def tapsFirst

	
		First portion of known taps (a combined map hits the 64KB JVM method limit) 
	


	.. chisel:attr:: private def tapsSecond

	
		Second portion of known taps (a combined map hits the 64KB JVM method limit) 
	


PRNG.scala
----------
.. chisel:attr:: class PRNGIO(val n: Int) extends Bundle

	Pseudo Random Number Generators (PRNG) interface	
	:param n: the width of the LFSR
	  

.. chisel:attr:: abstract class PRNG(val width: Int, val seed: Option[BigInt], step: Int = 1, updateSeed: Boolean = false) extends Module

	An abstract class representing a Pseudo Random Number Generator (PRNG)	
	:param width: the width of the PRNG
	
	:param seed: the initial state of the PRNG
	
	:param step: the number of state updates per cycle
	
	:param updateSeed: if true, when loading the seed the state will be updated as if the seed were the current state, if
		false, the state will be set to the seed
	  

.. chisel:attr:: abstract class PRNG(val width: Int, val seed: Option[BigInt], step: Int


	.. chisel:attr:: def delta(s: Seq[Bool]): Seq[Bool]

	
		State update function	
		:param s: input state
		:return: the next state
		    


	.. chisel:attr:: final def nextState(s: Seq[Bool]): Seq[Bool]

	
		The method that will be used to update the state of this PRNG	
		:param s: input state
		:return: the next state after `step` applications of :chisel:reref:`PRNG.delta`
		    


.. chisel:attr:: object PRNG

	Helper utilities related to the construction of Pseudo Random Number Generators (PRNGs) 


	.. chisel:attr:: def apply(gen: => PRNG, increment: Bool = true.B): UInt =

	
		Wrap a :chisel:reref:`PRNG`  to only return a pseudo-random :chisel:reref:`UInt` 	
		:param gen: a pseudo random number generator
		
		:param increment: when asserted the :chisel:reref:`PRNG`  will increment
		:return: the output (internal state) of the :chisel:reref:`PRNG`
		    


GaloisLFSR.scala
----------------
.. chisel:attr:: class GaloisLFSR(width: Int, taps: Set[Int], seed: Option[BigInt] = Some(1), val reduction: LFSRReduce = XOR, step: Int = 1, updateSeed: Boolean = false) extends PRNG(width, seed, step, updateSeed) with LFSR

	Galois Linear Feedback Shift Register (LFSR) generator.	
	A Galois LFSR can be generated by defining a width and a set of tap points. Optionally, an initial seed and a
	reduction operation (:chisel:reref:`XOR` , the default, or :chisel:reref:`XNOR` ) can be used to augment the generated hardware. The resulting
	hardware has support for a run-time programmable seed (via :chisel:reref:`PRNGIO.seed` ) and conditional increment (via
	:chisel:reref:`PRNGIO.increment` ).
	
	$seedExplanation
	
	In the example below, a 4-bit LFSR Fibonacci LFSR is constructed. The tap points are defined as four and three
	(using LFSR convention of indexing from one). This results in the hardware configuration shown in the diagram.
	
	
	.. code-block:: scala 

		 val lfsr4 = Module(new GaloisLFSR(4, Set(4, 3))
		 // +-----------------+---------------------------------------------------------+
		 // |                 |                                                         |
		 // |   +-------+     v     +-------+           +-------+           +-------+   |
		 // |   |       |   +---+   |       |           |       |           |       |   |
		 // +-->|  x^4  |-->|XOR|-->|  x^3  |---------->|  x^2  |---------->|  x^1  |---+
		 //     |       |   +---+   |       |           |       |           |       |
		 //     +-------+           +-------+           +-------+           +-------+
	
	
	If you require a maximal period Galois LFSR of a specific width, you can use :chisel:reref:`MaxPeriodGaloisLFSR` . If you only
	require a pseudorandom :chisel:reref:`UInt`  you can use the :chisel:reref:`GaloisLFSR$ GaloisLFSR companion object` .
	@see :chisel:reref:`https://en.wikipedia.org/wiki/Linear-feedback_shift_register#Galois_LFSRs`
	$paramWidth
	$paramTaps
	$paramSeed
	$paramReduction
	$paramStep
	$paramUpdateSeed
  

.. chisel:attr:: class GaloisLFSR(width: Int, taps: Set[Int], seed: Option[BigInt]


.. chisel:attr:: class MaxPeriodGaloisLFSR(width: Int, seed: Option[BigInt] = Some(1), reduction: LFSRReduce = XOR) extends GaloisLFSR(width, LFSR.tapsMaxPeriod.getOrElse(width, LFSR.badWidth(width)).head, seed, reduction)

	A maximal period Galois Linear Feedback Shift Register (LFSR) generator. The maximal period taps are sourced from	:chisel:reref:`LFSR.tapsMaxPeriod LFSR.tapsMaxPeriod` .
	
	.. code-block:: scala 

		 val lfsr8 = Module(new MaxPeriodGaloisLFSR(8))
	
	$paramWidth
	$paramSeed
	$paramReduction
  

.. chisel:attr:: class MaxPeriodGaloisLFSR(width: Int, seed: Option[BigInt]


.. chisel:attr:: object GaloisLFSR

	Utility for generating a pseudorandom :chisel:reref:`UInt`  from a :chisel:reref:`GaloisLFSR` .	
	For example, to generate a pseudorandom 8-bit :chisel:reref:`UInt`  that changes every cycle, you can use:
	
	.. code-block:: scala 

		 val pseudoRandomNumber = GaloisLFSR.maxPeriod(8)
	
	
	
@define paramWidth 	:param width: of pseudorandom output
	
@define paramTaps 	:param taps: a set of tap points to use when constructing the LFSR
	
@define paramIncrement 	:param increment: when asserted, a new random value will be generated
	
@define paramSeed 	:param seed: an initial value for internal LFSR state
	
@define paramReduction 	:param reduction: the reduction operation (either :chisel:reref:`XOR`  or
		:chisel:reref:`XNOR` )
	  

	.. chisel:attr:: def apply(width: Int, taps: Set[Int], increment: Bool = true.B, seed: Option[BigInt] = Some(1), reduction: LFSRReduce = XOR): UInt = PRNG(new GaloisLFSR(width, taps, seed, reduction), increment)

	
		Return a pseudorandom :chisel:reref:`UInt`  generated from a :chisel:reref:`FibonacciLFSR` .	$paramWidth
		$paramTaps
		$paramIncrement
		$paramSeed
		$paramReduction
	    


	.. chisel:attr:: def maxPeriod(width: Int, increment: Bool = true.B, seed: Option[BigInt] = Some(1), reduction: LFSRReduce = XOR): UInt = PRNG(new MaxPeriodGaloisLFSR(width, seed, reduction), increment)

	
		Return a pseudorandom :chisel:reref:`UInt`  generated using a maximal period :chisel:reref:`GaloisLFSR` 	$paramWidth
		$paramIncrement
		$paramSeed
		$paramReduction
	    


FibonacciLFSR.scala
-------------------
.. chisel:attr:: class FibonacciLFSR(width: Int, taps: Set[Int], seed: Option[BigInt] = Some(1), val reduction: LFSRReduce = XOR, step: Int = 1, updateSeed: Boolean = false) extends PRNG(width, seed, step, updateSeed) with LFSR

	Fibonacci Linear Feedback Shift Register (LFSR) generator.	
	A Fibonacci LFSR can be generated by defining a width and a set of tap points (corresponding to a polynomial). An
	optional initial seed and a reduction operation (:chisel:reref:`XOR` , the default, or :chisel:reref:`XNOR` ) can be used to augment the
	generated hardware. The resulting hardware has support for a run-time programmable seed (via :chisel:reref:`PRNGIO.seed` ) and
	conditional increment (via :chisel:reref:`PRNGIO.increment` ).
	
	$seedExplanation
	
	In the example below, a 4-bit Fibonacci LFSR is constructed. Tap points are defined as four and three (using LFSR
	convention of indexing from one). This results in the hardware configuration shown in the diagram.
	
	
	.. code-block:: scala 

		 val lfsr4 = Module(new FibonacciLFSR(4, Set(4, 3))
		 //                 +---+
		 // +-------------->|XOR|-------------------------------------------------------+
		 // |               +---+                                                       |
		 // |   +-------+     ^     +-------+           +-------+           +-------+   |
		 // |   |       |     |     |       |           |       |           |       |   |
		 // +---+  x^4  |<----+-----|  x^3  |<----------|  x^2  |<----------|  x^1  |<--+
		 //     |       |           |       |           |       |           |       |
		 //     +-------+           +-------+           +-------+           +-------+
	
	
	If you require a maximal period Fibonacci LFSR of a specific width, you can use :chisel:reref:`MaxPeriodFibonacciLFSR` . If you
	only require a pseudorandom :chisel:reref:`UInt`  you can use the :chisel:reref:`FibonacciLFSR$ FibonacciLFSR companion
	object` .
	@see :chisel:reref:`https://en.wikipedia.org/wiki/Linear-feedback_shift_register#Fibonacci_LFSRs`
	$paramWidth
	$paramTaps
	$paramSeed
	$paramReduction
	$paramStep
	$paramUpdateSeed
  

.. chisel:attr:: class FibonacciLFSR(width: Int, taps: Set[Int], seed: Option[BigInt]


.. chisel:attr:: class MaxPeriodFibonacciLFSR(width: Int, seed: Option[BigInt] = Some(1), reduction: LFSRReduce = XOR) extends FibonacciLFSR(width, LFSR.tapsMaxPeriod.getOrElse(width, LFSR.badWidth(width)).head, seed, reduction)

	A maximal period Fibonacci Linear Feedback Shift Register (LFSR) generator. The maximal period taps are sourced from	:chisel:reref:`LFSR.tapsMaxPeriod LFSR.tapsMaxPeriod` .
	
	.. code-block:: scala 

		 val lfsr8 = Module(new MaxPeriodFibonacciLFSR(8))
	
	$paramWidth
	$paramSeed
	$paramReduction
  

.. chisel:attr:: class MaxPeriodFibonacciLFSR(width: Int, seed: Option[BigInt]


.. chisel:attr:: object FibonacciLFSR

	Utility for generating a pseudorandom :chisel:reref:`UInt`  from a :chisel:reref:`FibonacciLFSR` .	
	For example, to generate a pseudorandom 8-bit :chisel:reref:`UInt`  that changes every cycle, you can use:
	
	.. code-block:: scala 

		 val pseudoRandomNumber = FibonacciLFSR.maxPeriod(8)
	
	
	
@define paramWidth 	:param width: of pseudorandom output
	
@define paramTaps 	:param taps: a set of tap points to use when constructing the LFSR
	
@define paramIncrement 	:param increment: when asserted, a new random value will be generated
	
@define paramSeed 	:param seed: an initial value for internal LFSR state
	
@define paramReduction 	:param reduction: the reduction operation (either :chisel:reref:`XOR`  or
		:chisel:reref:`XNOR` )
	  

	.. chisel:attr:: def apply(width: Int, taps: Set[Int], increment: Bool = true.B, seed: Option[BigInt] = Some(1), reduction: LFSRReduce = XOR): UInt = PRNG(new FibonacciLFSR(width, taps, seed, reduction), increment)

	
		Return a pseudorandom :chisel:reref:`UInt`  generated from a :chisel:reref:`FibonacciLFSR` .	$paramWidth
		$paramTaps
		$paramIncrement
		$paramSeed
		$paramReduction
	    


	.. chisel:attr:: def maxPeriod(width: Int, increment: Bool = true.B, seed: Option[BigInt] = Some(1), reduction: LFSRReduce = XOR): UInt = PRNG(new MaxPeriodFibonacciLFSR(width, seed, reduction), increment)

	
		Return a pseudorandom :chisel:reref:`UInt`  generated using a maximal period :chisel:reref:`FibonacciLFSR` 	$paramWidth
		$paramIncrement
		$paramSeed
		$paramReduction
	    


