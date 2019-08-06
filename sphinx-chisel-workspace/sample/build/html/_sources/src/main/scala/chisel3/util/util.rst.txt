---------------------------
src/main/scala/chisel3/util
---------------------------

.. toctree::
	experimental/experimental.rst
	random/random.rst


Mux.scala
---------
.. chisel:attr:: object Mux1H

	Builds a Mux tree out of the input signal vector using a one hot encoded	select signal. Returns the output of the Mux tree.
	
	
	.. code-block:: scala 

		 val hotValue = chisel3.util.Mux1H(Seq(
		  io.selector(0) -> 2.U,
		  io.selector(1) -> 4.U,
		  io.selector(2) -> 8.U,
		  io.selector(4) -> 11.U,
		 ))
	
	
	
	:note: results undefined if multiple select signals are simultaneously high
  

.. chisel:attr:: object PriorityMux

	Builds a Mux tree under the assumption that multiple select signals	can be enabled. Priority is given to the first select signal.
	
	
	.. code-block:: scala 

		 val hotValue = chisel3.util.PriorityMux(Seq(
		  io.selector(0) -> 2.U,
		  io.selector(1) -> 4.U,
		  io.selector(2) -> 8.U,
		  io.selector(4) -> 11.U,
		 ))
	
	Returns the output of the Mux tree.
  

.. chisel:attr:: object MuxLookup

	Creates a cascade of n Muxs to search for a key value.	
	
	.. code-block:: scala 

		 MuxLookup(idx, default,
		     Array(0.U -> a, 1.U -> b))
	
  

	.. chisel:attr:: def apply[S <: UInt, T <: Data] (key: S, default: T, mapping: Seq[(S, T)]): T =

	
		@param key a key to search for	
		:param default: a default value if nothing is found
		
		:param mapping: a sequence to search of keys and values
		:return: the value found or the default if not
		    


.. chisel:attr:: object MuxCase

	Given an association of values to enable signals, returns the first value with an associated	high enable signal.
	
	
	.. code-block:: scala 

		 MuxCase(default, Array(c1 -> a, c2 -> b))
	
  

	.. chisel:attr:: def apply[T <: Data] (default: T, mapping: Seq[(Bool, T)]): T =

	
		@param default the default value if none are enabled	
		:param mapping: a set of data values with associated enables
	    * @return the first value in mapping that is enabled 


Arbiter.scala
-------------
.. chisel:attr:: class ArbiterIO[T <: Data](private val gen: T, val n: Int) extends Bundle

	IO bundle definition for an Arbiter, which takes some number of ready-valid inputs and outputs	(selects) at most one.
	
	
	:param gen: data type
	
	:param n: number of inputs
	  

.. chisel:attr:: private object ArbiterCtrl

	Arbiter Control determining which producer has access  

.. chisel:attr:: abstract class LockingArbiterLike[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool]) extends Module


.. chisel:attr:: class LockingRRArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool]


.. chisel:attr:: class LockingArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool]


.. chisel:attr:: class RRArbiter[T <: Data](gen:T, n: Int) extends LockingRRArbiter[T](gen, n, 1)

	Hardware module that is used to sequence n producers into 1 consumer.	Producers are chosen in round robin order.
	
	
	:param gen: data type
	
	:param n: number of inputs
	
	.. code-block:: scala 

		 val arb = Module(new RRArbiter(UInt(), 2))
		 arb.io.in(0) <> producer0.io.out
		 arb.io.in(1) <> producer1.io.out
		 consumer.io.in <> arb.io.out
	
  

.. chisel:attr:: class Arbiter[T <: Data](gen: T, n: Int) extends Module

	Hardware module that is used to sequence n producers into 1 consumer.	Priority is given to lower producer.
	
	
	:param gen: data type
	
	:param n: number of inputs
		
	
	.. code-block:: scala 

		 val arb = Module(new Arbiter(UInt(), 2))
		 arb.io.in(0) <> producer0.io.out
		 arb.io.in(1) <> producer1.io.out
		 consumer.io.in <> arb.io.out
	
  

Reg.scala
---------
.. chisel:attr:: object RegEnable


	.. chisel:attr:: def apply[T <: Data](next: T, enable: Bool): T =

	
		Returns a register with the specified next, update enable gate, and no reset initialization.	
		
		.. code-block:: scala 
	
			 val regWithEnable = RegEnable(nextVal, ena)
		
	    


	.. chisel:attr:: def apply[T <: Data](next: T, init: T, enable: Bool): T =

	
		Returns a register with the specified next, update enable gate, and reset initialization.	
		
		.. code-block:: scala 
	
			 val regWithEnableAndReset = RegEnable(nextVal, 0.U, ena)
		
	    


.. chisel:attr:: object ShiftRegister


	.. chisel:attr:: def apply[T <: Data](in: T, n: Int, en: Bool = true.B): T =

	
		Returns the n-cycle delayed version of the input signal.	
		
		:param in: input to delay
		
		:param n: number of cycles to delay
		
		:param en: enable the shift
			
		
		.. code-block:: scala 
	
			 val regDelayTwo = ShiftRegister(nextVal, 2, ena)
		
	    


	.. chisel:attr:: def apply[T <: Data](in: T, n: Int, resetData: T, en: Bool): T =

	
		Returns the n-cycle delayed version of the input signal with reset initialization.	
		
		:param in: input to delay
		
		:param n: number of cycles to delay
		
		:param resetData: reset value for each register in the shift
		
		:param en: enable the shift
			
		
		.. code-block:: scala 
	
			 val regDelayTwoReset = ShiftRegister(nextVal, 2, 0.U, ena)
		
	    


BitPat.scala
------------
.. chisel:attr:: object BitPat


	.. chisel:attr:: private def parse(x: String): (BigInt, BigInt, Int) =

	
		Parses a bit pattern string into (bits, mask, width).	
		:return: bits the literal value, with don't cares being 0
		:return: mask the mask bits, with don't cares being 0 and cares being 1
		:return: width the number of bits in the literal, including values and
			don't cares.
		    


	.. chisel:attr:: def apply(n: String): BitPat =

	
		Creates a :chisel:reref:`BitPat`  literal from a string.	
		
		:param n: the literal value as a string, in binary, prefixed with 'b'
		
		:note: legal characters are '0', '1', and '?', as well as '_' and white
			space (which are ignored)
	    


	.. chisel:attr:: def dontCare(width: Int): BitPat

	
		Creates a :chisel:reref:`BitPat`  of all don't cares of the specified bitwidth.	
		
		.. code-block:: scala 
	
			 val myDontCare = BitPat.dontCare(4)  // equivalent to BitPat("b????")
		
	    


	.. chisel:attr:: def bitPatToUInt(x: BitPat): UInt =

	
		Allows BitPats to be used where a UInt is expected.	
		
		:note: the BitPat must not have don't care bits (will error out otherwise)
	    


	.. chisel:attr:: def apply(x: UInt): BitPat =

	
		Allows UInts to be used where a BitPat is expected, useful for when an	interface is defined with BitPats but not all cases need the partial
		matching capability.
		
		
		:note: the UInt must be a literal
	    


	.. chisel:attr:: def do_=== (that: BitPat) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_=/= (that: BitPat) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: sealed class BitPat(val value: BigInt, val mask: BigInt, width: Int) extends SourceInfoDoc

	Bit patterns are literals with masks, used to represent values with don't	care bits. Equality comparisons will ignore don't care bits.
	
	
	.. code-block:: scala 

		 "b10101".U === BitPat("b101??") // evaluates to true.B
		 "b10111".U === BitPat("b101??") // evaluates to true.B
		 "b10001".U === BitPat("b101??") // evaluates to false.B
	
  

	.. chisel:attr:: def do_=== (that: UInt) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_=/= (that: UInt) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =

	
		@group SourceInfoTransformMacro 
	


LFSR.scala
----------
.. chisel:attr:: object LFSR16

	LFSR16 generates a 16-bit linear feedback shift register, returning the register contents.	This is useful for generating a pseudo-random sequence.
	
	The example below, taken from the unit tests, creates two 4-sided dice using `LFSR16` primitives:
	
	.. code-block:: scala 

		   val bins = Reg(Vec(8, UInt(32.W)))
		
		   // Create two 4 sided dice and roll them each cycle.
		   // Use tap points on each LFSR so values are more independent
		   val die0 = Cat(Seq.tabulate(2) { i => LFSR16()(i) })
		   val die1 = Cat(Seq.tabulate(2) { i => LFSR16()(i + 2) })
		
		   val rollValue = die0 +& die1  // Note +& is critical because sum will need an extra bit.
		
		   bins(rollValue) := bins(rollValue) + 1.U
		
	
  

	.. chisel:attr:: def apply(increment: Bool = true.B): UInt = VecInit( FibonacciLFSR .maxPeriod(16, increment, seed = Some(BigInt(1) << 15)) .asBools .reverse ) .asUInt

	
		Generates a 16-bit linear feedback shift register, returning the register contents.	
		:param increment: optional control to gate when the LFSR updates.
		    


Cat.scala
---------
.. chisel:attr:: object Cat

	Concatenates elements of the input, in order, together.	
	
	.. code-block:: scala 

		 Cat("b101".U, "b11".U)  // equivalent to "b101 11".U
		 Cat(myUIntWire0, myUIntWire1)
		
		 Cat(Seq("b101".U, "b11".U))  // equivalent to "b101 11".U
		 Cat(mySeqOfBits)
	
  

	.. chisel:attr:: def apply[T <: Bits](a: T, r: T*): UInt

	
		Concatenates the argument data elements, in argument order, together. The first argument	forms the most significant bits, while the last argument forms the least significant bits.
	    


	.. chisel:attr:: def apply[T <: Bits](r: Seq[T]): UInt

	
		Concatenates the data elements of the input sequence, in reverse sequence order, together.	The first element of the sequence forms the most significant bits, while the last element
		in the sequence forms the least significant bits.
		
		Equivalent to r(0) ## r(1) ## ... ## r(n-1).
	    


BlackBoxUtils.scala
-------------------
.. chisel:attr:: trait HasBlackBoxResource extends BlackBox


	.. chisel:attr:: def addResource(blackBoxResource: String): Unit =

	
		Copies a resource file to the target directory	
		Resource files are located in project_root/src/main/resources/.
		Example of adding the resource file project_root/src/main/resources/blackbox.v:
		
		.. code-block:: scala 
	
			 addResource("/blackbox.v")
		
	    


.. chisel:attr:: trait HasBlackBoxInline extends BlackBox


.. chisel:attr:: trait HasBlackBoxPath extends BlackBox


	.. chisel:attr:: def addPath(blackBoxPath: String): Unit =

	
		Copies a file to the target directory	
		This works with absolute and relative paths. Relative paths are relative
		to the current working directory, which is generally not the same as the
		target directory.
	    


TransitName.scala
-----------------
.. chisel:attr:: object TransitName


	.. chisel:attr:: def apply[T<:HasId](from: T, to: HasId): T =

	
		Transit a name from one type to another	
		:param from: the thing with a "good" name
		
		:param to: the thing that will receive the "good" name
		:return: the `from` parameter
		    


	.. chisel:attr:: def withSuffix[T<:HasId](suffix: String)(from: T, to: HasId): T =

	
		Transit a name from one type to another ''and add a suffix''	
		:param suffix: the suffix to append
		
		:param from: the thing with a "good" name
		
		:param to: the thing that will receive the "good" name
		:return: the `from` parameter
		    


Enum.scala
----------
.. chisel:attr:: trait Enum

	Defines a set of unique UInt constants	
	Unpack with a list to specify an enumeration. Usually used with :chisel:reref:`switch`  to describe a finite
	state machine.
	
	
	.. code-block:: scala 

		 val state_on :: state_off :: Nil = Enum(2)
		 val current_state = WireDefault(state_off)
		 switch (current_state) {
		   is (state_on) {
		     ...
		   }
		   is (state_off) {
		     ...
		   }
		 }
	
  

	.. chisel:attr:: protected def createValues(n: Int): Seq[UInt]

	
		Returns a sequence of Bits subtypes with values from 0 until n. Helper method. 
	


	.. chisel:attr:: def apply(n: Int): List[UInt]

	
		Returns n unique UInt values	
		
		:param n: Number of unique UInt constants to enumerate
		:return: Enumerated constants
		    


.. chisel:attr:: object Enum extends Enum  


OneHot.scala
------------
.. chisel:attr:: object OHToUInt

	Returns the bit position of the sole high bit of the input bitvector.	
	Inverse operation of :chisel:reref:`UIntToOH` .
	
	
	.. code-block:: scala 

		 OHToUInt("b0100".U) // results in 2.U
	
	
	
	:note: assumes exactly one high bit, results undefined otherwise
  

.. chisel:attr:: object PriorityEncoder

	Returns the bit position of the least-significant high bit of the input bitvector.	
	
	.. code-block:: scala 

		 PriorityEncoder("b0110".U) // results in 1.U
	
	
	Multiple bits may be high in the input.
  

.. chisel:attr:: object UIntToOH

	Returns the one hot encoding of the input UInt.	
	
	.. code-block:: scala 

		 UIntToOH(2.U) // results in "b0100".U
	
	
  

.. chisel:attr:: object PriorityEncoderOH

	Returns a bit vector in which only the least-significant 1 bit in the input vector, if any,	is set.
	
	
	.. code-block:: scala 

		 PriorityEncoderOH((false.B, true.B, true.B, false.B)) // results in (false.B, false.B, true.B, false.B)
	
  

util.scala
----------
.. chisel:attr:: package object util

	The util package provides extensions to core chisel for common hardware components and utility	functions
  

Valid.scala
-----------
.. chisel:attr:: class Valid[+T <: Data](gen: T) extends Bundle

	A :chisel:reref:`Bundle`  that adds a `valid` bit to some data. This indicates that the user expects a "valid" interface between	a producer and a consumer. Here, the producer asserts the `valid` bit when data on the `bits` line contains valid
	data. This differs from :chisel:reref:`DecoupledIO`  or :chisel:reref:`IrrevocableIO`  as there is no `ready` line that the consumer can use
	to put back pressure on the producer.
	
	In most scenarios, the `Valid` class will ''not'' be used directly. Instead, users will create `Valid` interfaces
	using the :chisel:reref:`Valid$ Valid factory` .
	
	:type-param T: the type of the data
	
	:param gen: some data
		@see :chisel:reref:`Valid$ Valid factory`  for concrete examples
	  

	.. chisel:attr:: def fire(dummy: Int = 0): Bool = valid

	
		True when `valid` is asserted	:return: a Chisel :chisel:reref:`Bool`  true if `valid` is asserted
		    


.. chisel:attr:: object Valid

	Factory for generating "valid" interfaces. A "valid" interface is a data-communicating interface between a producer	and a consumer where the producer does not wait for the consumer. Concretely, this means that one additional bit is
	added to the data indicating its validity.
	
	As an example, consider the following :chisel:reref:`Bundle` , `MyBundle`:
	
	.. code-block:: scala 

		   class MyBundle extends Bundle {
		     val foo = Output(UInt(8.W))
		   }
	
	
	To convert this to a "valid" interface, you wrap it with a call to the :chisel:reref:`Valid$.apply `Valid` companion object's
	apply method` :
	
	.. code-block:: scala 

		   val bar = Valid(new MyBundle)
	
	
	The resulting interface is ''structurally'' equivalent to the following:
	
	.. code-block:: scala 

		   class MyValidBundle extends Bundle {
		     val valid = Output(Bool())
		     val bits = Output(new MyBundle)
		   }
	
	
	In addition to adding the `valid` bit, a :chisel:reref:`Valid.fire`  method is also added that returns the `valid` bit. This
	provides a similarly named interface to :chisel:reref:`DecoupledIO` 's fire.
	
	@see :chisel:reref:`Decoupled$ DecoupledIO Factory`
	@see :chisel:reref:`Irrevocable$ IrrevocableIO Factory`
  

	.. chisel:attr:: def apply[T <: Data](gen: T): Valid[T]

	
		Wrap some :chisel:reref:`Data`  in a valid interface	
		:type-param T: the type of the data to wrap
		
		:param gen: the data to wrap
		:return: the wrapped input data
		    


.. chisel:attr:: object Pipe


	.. chisel:attr:: def apply[T <: Data](enqValid: Bool, enqBits: T, latency: Int)(implicit compileOptions: CompileOptions): Valid[T] =

	
		Generate a pipe from an explicit valid bit and some data	
		:param enqValid: the valid bit (must be a hardware type)
		
		:param enqBits: the data (must be a hardware type)
		
		:param latency: the number of pipeline stages
		:return: $returnType
		    


	.. chisel:attr:: def apply[T <: Data](enqValid: Bool, enqBits: T)(implicit compileOptions: CompileOptions): Valid[T] =

	
		Generate a one-stage pipe from an explicit valid bit and some data	
		:param enqValid: the valid bit (must be a hardware type)
		
		:param enqBits: the data (must be a hardware type)
		:return: $returnType
		    


	.. chisel:attr:: def apply[T <: Data](enq: Valid[T], latency: Int = 1)(implicit compileOptions: CompileOptions): Valid[T] =

	
		Generate a pipe for a :chisel:reref:`Valid`  interface	
		:param enq: a :chisel:reref:`Valid`  interface (must be a hardware type)
		
		:param latency: the number of pipeline stages
		:return: $returnType
		    


.. chisel:attr:: class Pipe[T <: Data](gen: T, latency: Int = 1)(implicit compileOptions: CompileOptions) extends Module

	Pipeline module generator parameterized by data type and latency.	
	This defines a module with one input, `enq`, and one output, `deq`. The input and output are :chisel:reref:`Valid`  interfaces
	that wrap some Chisel type, e.g., a :chisel:reref:`UInt`  or a :chisel:reref:`Bundle` . This generator will then chain together a number of
	pipeline stages that all advance when the input :chisel:reref:`Valid`  `enq` fires. The output `deq` :chisel:reref:`Valid`  will fire only
	when valid data has made it all the way through the pipeline.
	
	As an example, to construct a 4-stage pipe of 8-bit :chisel:reref:`UInt` s and connect it to a producer and consumer, you can use
	the following:
	
	.. code-block:: scala 

		   val foo = Module(new Pipe(UInt(8.W)), 4)
		   pipe.io.enq := producer.io
		   consumer.io := pipe.io.deq
	
	
	If you already have the :chisel:reref:`Valid`  input or the components of a :chisel:reref:`Valid`  interface, it may be simpler to use the
	:chisel:reref:`Pipe$ Pipe factory`  companion object. This, which :chisel:reref:`Pipe`  internally utilizes, will automatically connect the
	input for you.
	
	
	:param gen: a Chisel type
	
	:param latency: the number of pipeline stages
		@see :chisel:reref:`Pipe$ Pipe factory`  for an alternative API
		@see :chisel:reref:`Valid`  interface
		@see :chisel:reref:`Queue`  and the :chisel:reref:`Queue$ Queue factory`  for actual queues
		@see The :chisel:reref:`ShiftRegister$ ShiftRegister factory`  to generate a pipe without a :chisel:reref:`Valid`  interface
	  

.. chisel:attr:: class Pipe[T <: Data](gen: T, latency: Int


.. chisel:attr:: class PipeIO extends Bundle

	Interface for :chisel:reref:`Pipe` s composed of a :chisel:reref:`Valid`  input and :chisel:reref:`Valid`  output	@define notAQueue
    

Bitwise.scala
-------------
.. chisel:attr:: object FillInterleaved

	Creates repetitions of each bit of the input in order.	
	
	.. code-block:: scala 

		 FillInterleaved(2, "b1 0 0 0".U)  // equivalent to "b11 00 00 00".U
		 FillInterleaved(2, "b1 0 0 1".U)  // equivalent to "b11 00 00 11".U
		 FillInterleaved(2, myUIntWire)  // dynamic interleaved fill
		
		 FillInterleaved(2, Seq(true.B, false.B, false.B, false.B))  // equivalent to "b11 00 00 00".U
		 FillInterleaved(2, Seq(true.B, false.B, false.B, true.B))  // equivalent to "b11 00 00 11".U
	
  

	.. chisel:attr:: def apply(n: Int, in: UInt): UInt

	
		Creates n repetitions of each bit of x in order.	
		Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
	    


	.. chisel:attr:: def apply(n: Int, in: Seq[Bool]): UInt

	
		Creates n repetitions of each bit of x in order.	
		Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
	    


.. chisel:attr:: object PopCount

	Returns the number of bits set (value is 1 or true) in the input signal.	
	
	.. code-block:: scala 

		 PopCount(Seq(true.B, false.B, true.B, true.B))  // evaluates to 3.U
		 PopCount(Seq(false.B, false.B, true.B, false.B))  // evaluates to 1.U
		
		 PopCount("b1011".U)  // evaluates to 3.U
		 PopCount("b0010".U)  // evaluates to 1.U
		 PopCount(myUIntWire)  // dynamic count
	
  

.. chisel:attr:: object Fill

	Create repetitions of the input using a tree fanout topology.	
	
	.. code-block:: scala 

		 Fill(2, "b1000".U)  // equivalent to "b1000 1000".U
		 Fill(2, "b1001".U)  // equivalent to "b1001 1001".U
		 Fill(2, myUIntWire)  // dynamic fill
	
  

	.. chisel:attr:: def apply(n: Int, x: UInt): UInt =

	
		Create n repetitions of x using a tree fanout topology.	
		Output data-equivalent to x ## x ## ... ## x (n repetitions).
	    


.. chisel:attr:: object Reverse

	Returns the input in bit-reversed order. Useful for little/big-endian conversion.	
	
	.. code-block:: scala 

		 Reverse("b1101".U)  // equivalent to "b1011".U
		 Reverse("b1101".U(8.W))  // equivalent to "b10110000".U
		 Reverse(myUIntWire)  // dynamic reverse
	
  

Conditional.scala
-----------------
.. chisel:attr:: object unless


	.. chisel:attr:: def apply(c: Bool)(block: => Unit)

	
		Does the same thing as :chisel:reref:`when$ when` , but with the condition inverted.    


.. chisel:attr:: class SwitchContext[T <: Element](cond: T, whenContext: Option[WhenContext], lits: Set[BigInt])

	Implementation details for :chisel:reref:`switch` . See :chisel:reref:`switch`  and :chisel:reref:`chisel3.util.is is`  for the	user-facing API.
	
	:note: DO NOT USE. This API is subject to change without warning.
  

.. chisel:attr:: object is

	Use to specify cases in a :chisel:reref:`switch`  block, equivalent to a :chisel:reref:`when$ when`  block comparing to	the condition variable.
	
	
	:note: illegal outside a :chisel:reref:`switch`  block
	
	:note: must be a literal
	
	:note: each is must be mutually exclusive
	
	:note: dummy implementation, a macro inside :chisel:reref:`switch`  transforms this into the actual
		implementation
  

	.. chisel:attr:: def apply(v: Iterable[Element])(block: => Unit)

	
		Executes `block` if the switch condition is equal to any of the values in `v`.    


	.. chisel:attr:: def apply(v: Element)(block: => Unit)

	
		Executes `block` if the switch condition is equal to `v`.    


	.. chisel:attr:: def apply(v: Element, vr: Element*)(block: => Unit)

	
		Executes `block` if the switch condition is equal to any of the values in the argument list.    


.. chisel:attr:: object switch

	Conditional logic to form a switch block. See :chisel:reref:`is$ is`  for the case API.	
	
	.. code-block:: scala 

		 switch (myState) {
		   is (state1) {
		     // some logic here that runs when myState === state1
		   }
		   is (state2) {
		     // some logic here that runs when myState === state2
		   }
		 }
	
  

Lookup.scala
------------
.. chisel:attr:: object ListLookup

	For each element in a list, muxes (looks up) between cases (one per list element) based on a	common address.
	
	
	:note: This appears to be an odd, specialized operator that we haven't seen used much, and seems
		to be a holdover from chisel2. This may be deprecated and removed, usage is not
		recommended.
		
	
	:param addr: common select for cases, shared (same) across all list elements
	
	:param default: default value for each list element, should the address not match any case
	
	:param mapping: list of cases, where each entry consists of a :chisel:reref:`chisel3.util.BitPat BitPath`  (compared against addr) and
		a list of elements (same length as default) that is the output value for that
		element (will have the same index in the output).
		
	
	.. code-block:: scala 

		 ListLookup(2.U,  // address for comparison
		                          List(10.U, 11.U, 12.U),   // default "row" if none of the following cases match
		     Array(BitPat(2.U) -> List(20.U, 21.U, 22.U),  // this "row" hardware-selected based off address 2.U
		           BitPat(3.U) -> List(30.U, 31.U, 32.U))
		 ) // hardware-evaluates to List(20.U, 21.U, 22.U)
		 // Note: if given address 0.U, the above would hardware evaluate to List(10.U, 11.U, 12.U)
	
  

.. chisel:attr:: object Lookup

	Muxes between cases based on whether an address matches any pattern for a case.	Similar to :chisel:reref:`chisel3.util.MuxLookup MuxLookup` , but uses :chisel:reref:`chisel3.util.BitPat BitPat`  for address comparison.
	
	
	:note: This appears to be an odd, specialized operator that we haven't seen used much, and seems
		to be a holdover from chisel2. This may be deprecated and removed, usage is not
		recommended.
		
	
	:param addr: address to select between cases
	
	:param default: default value should the address not match any case
	
	:param mapping: list of cases, where each entry consists of a :chisel:reref:`chisel3.util.BitPat BitPat`  (compared against addr) and the
		output value if the BitPat matches
	  

Math.scala
----------
.. chisel:attr:: object log2Up

	Compute the log2 of a Scala integer, rounded up, with min value of 1.	Useful for getting the number of bits needed to represent some number of states (in - 1),
	To get the number of bits needed to represent some number n, use log2Up(n + 1).
	with the minimum value preventing the creation of currently-unsupported zero-width wires.
	
	Note: prefer to use log2Ceil when in is known to be > 1 (where log2Ceil(in) > 0).
	This will be deprecated when zero-width wires is supported.
	
	
	.. code-block:: scala 

		 log2Up(1)  // returns 1
		 log2Up(2)  // returns 1
		 log2Up(3)  // returns 2
		 log2Up(4)  // returns 2
	
  

.. chisel:attr:: object log2Ceil

	Compute the log2 of a Scala integer, rounded up.	Useful for getting the number of bits needed to represent some number of states (in - 1).
	To get the number of bits needed to represent some number n, use log2Ceil(n + 1).
	
	Note: can return zero, and should not be used in cases where it may generate unsupported
	zero-width wires.
	
	
	.. code-block:: scala 

		 log2Ceil(1)  // returns 0
		 log2Ceil(2)  // returns 1
		 log2Ceil(3)  // returns 2
		 log2Ceil(4)  // returns 2
	
  

.. chisel:attr:: object log2Down

	Compute the log2 of a Scala integer, rounded down, with min value of 1.	
	
	.. code-block:: scala 

		 log2Down(1)  // returns 1
		 log2Down(2)  // returns 1
		 log2Down(3)  // returns 1
		 log2Down(4)  // returns 2
	
  

.. chisel:attr:: object log2Floor

	Compute the log2 of a Scala integer, rounded down.	
	Can be useful in computing the next-smallest power of two.
	
	
	.. code-block:: scala 

		 log2Floor(1)  // returns 0
		 log2Floor(2)  // returns 1
		 log2Floor(3)  // returns 1
		 log2Floor(4)  // returns 2
	
  

.. chisel:attr:: object isPow2

	Returns whether a Scala integer is a power of two.	
	
	.. code-block:: scala 

		 isPow2(1)  // returns true
		 isPow2(2)  // returns true
		 isPow2(3)  // returns false
		 isPow2(4)  // returns true
	
  

.. chisel:attr:: object unsignedBitLength


	.. chisel:attr:: def apply(in: BigInt): Int =

	
		Return the number of bits required to encode a specific value, assuming no sign bit is required.	
		Basically, `n.bitLength`. NOTE: This will return 0 for a value of 0.
		This reflects the Chisel assumption that a zero width wire has a value of 0.
		
		:param in: - the number to be encoded.
		:return: - an Int representing the number of bits to encode.
		    


.. chisel:attr:: object signedBitLength


	.. chisel:attr:: def apply(in: BigInt): Int =

	
		Return the number of bits required to encode a specific value, assuming a sign bit is required.	
		Basically, 0 for 0, 1 for -1, and `n.bitLength` + 1 for everything else.
		This reflects the Chisel assumption that a zero width wire has a value of 0.
		
		:param in: - the number to be encoded.
		:return: - an Int representing the number of bits to encode.
		    


Decoupled.scala
---------------
.. chisel:attr:: abstract class ReadyValidIO[+T <: Data](gen: T) extends Bundle

	An I/O Bundle containing 'valid' and 'ready' signals that handshake	the transfer of data stored in the 'bits' subfield.
	The base protocol implied by the directionality is that
	the producer uses the interface as-is (outputs bits)
	while the consumer uses the flipped interface (inputs bits).
	The actual semantics of ready/valid are enforced via the use of concrete subclasses.
	
	:param gen: the type of data to be wrapped in Ready/Valid
	  

.. chisel:attr:: object ReadyValidIO


	.. chisel:attr:: def fire(): Bool

	
		Indicates if IO is both ready and valid     


	.. chisel:attr:: def enq(dat: T): T =

	
		Push dat onto the output bits of this interface to let the consumer know it has happened.	
		:param dat: the values to assign to bits.
		:return:    dat.
		      


	.. chisel:attr:: def noenq(): Unit =

	
		Indicate no enqueue occurs. Valid is set to false, and bits are	connected to an uninitialized wire.
	      


	.. chisel:attr:: def deq(): T =

	
		Assert ready on this port and return the associated data bits.	This is typically used when valid has been asserted by the producer side.
		:return: The data bits.
		      


	.. chisel:attr:: def nodeq(): Unit =

	
		Indicate no dequeue occurs. Ready is set to false.      


.. chisel:attr:: class DecoupledIO[+T <: Data](gen: T) extends ReadyValidIO[T](gen)

	A concrete subclass of ReadyValidIO signaling that the user expects a	"decoupled" interface: 'valid' indicates that the producer has
	put valid data in 'bits', and 'ready' indicates that the consumer is ready
	to accept the data this cycle. No requirements are placed on the signaling
	of ready or valid.
	
	:param gen: the type of data to be wrapped in DecoupledIO
	  

.. chisel:attr:: object Decoupled

	This factory adds a decoupled handshaking protocol to a data bundle. 


	.. chisel:attr:: def apply[T <: Data](gen: T): DecoupledIO[T]

	
		Wraps some Data with a DecoupledIO interface. 
	


	.. chisel:attr:: def apply[T <: Data](irr: IrrevocableIO[T]): DecoupledIO[T] =

	
		Downconverts an IrrevocableIO output to a DecoupledIO, dropping guarantees of irrevocability.	
		
		:note: unsafe (and will error) on the producer (input) side of an IrrevocableIO
	    


.. chisel:attr:: class IrrevocableIO[+T <: Data](gen: T) extends ReadyValidIO[T](gen)

	A concrete subclass of ReadyValidIO that promises to not change	the value of 'bits' after a cycle where 'valid' is high and 'ready' is low.
	Additionally, once 'valid' is raised it will never be lowered until after
	'ready' has also been raised.
	
	:param gen: the type of data to be wrapped in IrrevocableIO
	  

.. chisel:attr:: object Irrevocable

	Factory adds an irrevocable handshaking protocol to a data bundle. 


	.. chisel:attr:: def apply[T <: Data](dec: DecoupledIO[T]): IrrevocableIO[T] =

	
		Upconverts a DecoupledIO input to an IrrevocableIO, allowing an IrrevocableIO to be used	where a DecoupledIO is expected.
		
		
		:note: unsafe (and will error) on the consumer (output) side of an DecoupledIO
	    


.. chisel:attr:: object EnqIO

	Producer - drives (outputs) valid and bits, inputs ready.	
	:param gen: The type of data to enqueue
	  

.. chisel:attr:: object DeqIO

	Consumer - drives (outputs) ready, inputs valid and bits.	
	:param gen: The type of data to dequeue
	  

.. chisel:attr:: class QueueIO[T <: Data](private val gen: T, val entries: Int) extends Bundle

	An I/O Bundle for Queues	
	:param gen: The type of data to queue
	
	:param entries: The max number of entries in the queue.
	  

.. chisel:attr:: class Queue[T <: Data](gen: T, val entries: Int, pipe: Boolean = false, flow: Boolean = false) (implicit compileOptions: chisel3.CompileOptions) extends Module()

	A hardware module implementing a Queue	
	:param gen: The type of data to queue
	
	:param entries: The max number of entries in the queue
	
	:param pipe: True if a single entry queue can run at full throughput (like a pipeline). The ''ready'' signals are
		combinationally coupled.
	
	:param flow: True if the inputs can be consumed on the same cycle (the inputs "flow" through the queue immediately).
		The ''valid'' signals are coupled.
		
	
	.. code-block:: scala 

		 val q = Module(new Queue(UInt(), 16))
		 q.io.enq <> producer.io.out
		 consumer.io.in <> q.io.deq
	
  

.. chisel:attr:: class Queue[T <: Data](gen: T, val entries: Int, pipe: Boolean


.. chisel:attr:: object Queue

	Factory for a generic hardware queue.	
	
	:param enq: input (enqueue) interface to the queue, also determines width of queue elements
	
	:param entries: depth (number of elements) of the queue
		
	:return: output (dequeue) interface from the queue
		
	
	.. code-block:: scala 

		 consumer.io.in <> Queue(producer.io.out, 16)
	
  

	.. chisel:attr:: def apply[T <: Data](enq: ReadyValidIO[T], entries: Int = 2, pipe: Boolean = false, flow: Boolean = false): DecoupledIO[T] =

	
		Create a queue and supply a DecoupledIO containing the product. 
	


	.. chisel:attr:: def irrevocable[T <: Data](enq: ReadyValidIO[T], entries: Int = 2, pipe: Boolean = false, flow: Boolean = false): IrrevocableIO[T] =

	
		Create a queue and supply a IrrevocableIO containing the product.	Casting from Decoupled is safe here because we know the Queue has
		Irrevocable semantics; we didn't want to change the return type of
		apply() for backwards compatibility reasons.
	    


CircuitMath.scala
-----------------
.. chisel:attr:: object Log2

	Returns the base-2 integer logarithm of an UInt.	
	
	:note: The result is truncated, so e.g. Log2(13.U) === 3.U
		
	
	.. code-block:: scala 

		 Log2(8.U)  // evaluates to 3.U
		 Log2(13.U)  // evaluates to 3.U (truncation)
		 Log2(myUIntWire)
	
	
  

	.. chisel:attr:: def apply(x: Bits, width: Int): UInt =

	
		Returns the base-2 integer logarithm of the least-significant `width` bits of an UInt.    


MixedVec.scala
--------------
.. chisel:attr:: object MixedVecInit

		Create a MixedVec wire with default values as specified, and type of each element inferred from
	those default values.
	
	This is analogous to :chisel:reref:`VecInit` .
	:return: MixedVec with given values assigned
		
	
	.. code-block:: scala 

		 MixedVecInit(Seq(100.U(8.W), 10000.U(16.W), 101.U(32.W)))
	
  

	.. chisel:attr:: def apply[T <: Data](vals: Seq[T]): MixedVec[T] =

	
			Create a MixedVec wire from a Seq of values.
	    


	.. chisel:attr:: def apply[T <: Data](val0: T, vals: T*): MixedVec[T]

	
			Create a MixedVec wire from a varargs list of values.
	    


.. chisel:attr:: object MixedVec

		Create a MixedVec type, given element types. Inputs must be Chisel types which have no value
	(not hardware types).
	
	:return: MixedVec with the given types.
	  

	.. chisel:attr:: def apply[T <: Data](eltsIn: Seq[T]): MixedVec[T]

	
			Create a MixedVec type from a Seq of Chisel types.
	    


	.. chisel:attr:: def apply[T <: Data](val0: T, vals: T*): MixedVec[T]

	
			Create a MixedVec type from a varargs list of Chisel types.
	    


	.. chisel:attr:: def apply[T <: Data](mixedVec: MixedVec[T]): MixedVec[T]

	
			Create a new MixedVec type from an unbound MixedVec type.
	    


	.. chisel:attr:: def apply[T <: Data](vec: Vec[T]): MixedVec[T] =

	
			Create a MixedVec type from the type of the given Vec.
		
		
		.. code-block:: scala 
	
			 MixedVec(Vec(2, UInt(8.W))) = MixedVec(Seq.fill(2){UInt(8.W)})
		
	    


.. chisel:attr:: final class MixedVec[T <: Data](private val eltsIn: Seq[T]) extends Record with collection.IndexedSeq[T]

		A hardware array of elements that can hold values of different types/widths,
	unlike Vec which can only hold elements of the same type/width.
	
	
	:param eltsIn: Element types. Must be Chisel types.
		
	
	.. code-block:: scala 

		 val v = Wire(MixedVec(Seq(UInt(8.W), UInt(16.W), UInt(32.W))))
		 v(0) := 100.U(8.W)
		 v(1) := 10000.U(16.W)
		 v(2) := 101.U(32.W)
	
  

	.. chisel:attr:: def apply(index: Int): T

	
			Statically (elaboration-time) retrieve the element at the given index.
		
		:param index: Index with which to retrieve.
		:return: Retrieved index.
		    


	.. chisel:attr:: def :=(that: Seq[T]): Unit =

	
		Strong bulk connect, assigning elements in this MixedVec from elements in a Seq.	
		
		:note: the lengths of this and that must match
	    


	.. chisel:attr:: def length: Int

	
			Get the length of this MixedVec.
		:return: Number of elements in this MixedVec.
		    


Counter.scala
-------------
.. chisel:attr:: class Counter(val n: Int)

	Used to generate an inline (logic directly in the containing Module, no internal Module is created)	hardware counter.
	
	Typically instantiated with apply methods in :chisel:reref:`Counter$ object Counter`
	
	Does not create a new Chisel Module
	
	
	.. code-block:: scala 

		   val countOn = true.B // increment counter every clock cycle
		   val (counterValue, counterWrap) = Counter(countOn, 4)
		   when (counterValue === 3.U) {
		     ...
		   }
	
	
	
	:param n: number of counts before the counter resets (or one more than the
		maximum output value of the counter), need not be a power of two
	  

	.. chisel:attr:: def inc(): Bool =

	
		Increment the counter, returning whether the counter currently is at the	maximum and will wrap. The incremented value is registered and will be
		visible on the next cycle.
	    


.. chisel:attr:: object Counter


	.. chisel:attr:: def apply(n: Int): Counter

	
		Instantiate a :chisel:reref:`Counter! counter`  with the specified number of counts.    


	.. chisel:attr:: def apply(cond: Bool, n: Int): (UInt, Bool) =

	
		Instantiate a :chisel:reref:`Counter! counter`  with the specified number of counts and a gate.	
		
		:param cond: condition that controls whether the counter increments this cycle
		
		:param n: number of counts before the counter resets
		:return: tuple of the counter value and whether the counter will wrap (the value is at
			maximum and the condition is true).
		    


ImplicitConversions.scala
-------------------------
.. chisel:attr:: object ImplicitConversions

	Implicit conversions to automatically convert :chisel:reref:`scala.Boolean`  and :chisel:reref:`scala.Int`  to :chisel:reref:`Bool` 	and :chisel:reref:`UInt`  respectively
  

