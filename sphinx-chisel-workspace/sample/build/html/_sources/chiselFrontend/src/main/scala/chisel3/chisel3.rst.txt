-------------------------------------
chiselFrontend/src/main/scala/chisel3
-------------------------------------

.. toctree::
	experimental/experimental.rst
	internal/internal.rst
	core/core.rst


Mux.scala
---------
.. chisel:attr:: object Mux extends SourceInfoDoc


	.. chisel:attr:: def apply[T <: Data](cond: Bool, con: T, alt: T): T

	
		Creates a mux, whose output is one of the inputs depending on the	value of the condition.
		
		
		:param cond: condition determining the input to choose
		
		:param con: the value chosen when `cond` is true
		
		:param alt: the value chosen when `cond` is false
			@example
		
		.. code-block:: scala 
	
			 val muxOut = Mux(data_in === 3.U, 3.U(4.W), 0.U(4.W))
		
	    


	.. chisel:attr:: def do_apply[T <: Data](cond: Bool, con: T, alt: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		@group SourceInfoTransformMacro 
	


When.scala
----------
.. chisel:attr:: object when


	.. chisel:attr:: def apply(cond: => Bool)(block: => Unit)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): WhenContext =

	
		Create a `when` condition block, where whether a block of logic is	executed or not depends on the conditional.
		
		
		:param cond: condition to execute upon
		
		:param block: logic that runs only if `cond` is true
			
			@example
		
		.. code-block:: scala 
	
			 when ( myData === 3.U ) {
			   // Some logic to run when myData equals 3.
			 } .elsewhen ( myData === 1.U ) {
			   // Some logic to run when myData equals 1.
			 } .otherwise {
			   // Some logic to run when myData is neither 3 nor 1.
			 }
		
	    


.. chisel:attr:: final class WhenContext(sourceInfo: SourceInfo, cond: Option[() => Bool], block: => Unit, firrtlDepth: Int = 0)

	A WhenContext may represent a when, and elsewhen, or an	otherwise. Since FIRRTL does not have an "elsif" statement,
	alternatives must be mapped to nested if-else statements inside
	the alternatives of the preceeding condition. In order to emit
	proper FIRRTL, it is necessary to keep track of the depth of
	nesting of the FIRRTL whens. Due to the "thin frontend" nature of
	Chisel3, it is not possible to know if a when or elsewhen has a
	succeeding elsewhen or otherwise; therefore, this information is
	added by preprocessing the command queue.
  

.. chisel:attr:: final class WhenContext(sourceInfo: SourceInfo, cond: Option[() => Bool], block: => Unit, firrtlDepth: Int


	.. chisel:attr:: def elsewhen (elseCond: => Bool)(block: => Unit)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): WhenContext =

	
		This block of logic gets executed if above conditions have been	false and this condition is true. The lazy argument pattern
		makes it possible to delay evaluation of cond, emitting the
		declaration and assignment of the Bool node of the predicate in
		the correct place.
	    


	.. chisel:attr:: def otherwise(block: => Unit)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit

	
		This block of logic gets executed only if the above conditions	were all false. No additional logic blocks may be appended past
		the `otherwise`. The lazy argument pattern makes it possible to
		delay evaluation of cond, emitting the declaration and
		assignment of the Bool node of the predicate in the correct
		place.
	    


Data.scala
----------
.. chisel:attr:: sealed abstract class SpecifiedDirection object SpecifiedDirection

	User-specified directions.  

.. chisel:attr:: object SpecifiedDirection


.. chisel:attr:: case object Unspecified extends SpecifiedDirection

	Default user direction, also meaning 'not-flipped'    

.. chisel:attr:: case object Output extends SpecifiedDirection

	Node and its children are forced as output    

.. chisel:attr:: case object Input extends SpecifiedDirection

	Node and its children are forced as inputs    

.. chisel:attr:: case object Flip extends SpecifiedDirection

	Mainly for containers, children are flipped.    

	.. chisel:attr:: def fromParent(parentDirection: SpecifiedDirection, thisDirection: SpecifiedDirection): SpecifiedDirection

	
		Returns the effective SpecifiedDirection of this node given the parent's effective SpecifiedDirection	and the user-specified SpecifiedDirection of this node.
	    


.. chisel:attr:: sealed abstract class ActualDirection

	Resolved directions for both leaf and container nodes, only visible after	a node is bound (since higher-level specifications like Input and Output
	can override directions).
  

.. chisel:attr:: sealed abstract class ActualDirection  object ActualDirection


.. chisel:attr:: object ActualDirection


.. chisel:attr:: case object Empty extends ActualDirection

	The object does not exist / is empty and hence has no direction    

.. chisel:attr:: case object Unspecified extends ActualDirection

	Undirectioned, struct-like    

.. chisel:attr:: case object Output extends ActualDirection

	Output element, or container with all outputs (even if forced)    

.. chisel:attr:: case object Input extends ActualDirection

	Input element, or container with all inputs (even if forced)    

	.. chisel:attr:: def fromChildren(childDirections: Set[ActualDirection], containerDirection: SpecifiedDirection): Option[ActualDirection] =

	
		Determine the actual binding of a container given directions of its children.	Returns None in the case of mixed specified / unspecified directionality.
	    


.. chisel:attr:: object DataMirror

	Experimental hardware construction reflection API    

.. chisel:attr:: private[chisel3] object cloneSupertype

	Creates a clone of the super-type of the input elements. Super-type is defined as:	- for Bits type of the same class: the cloned type of the largest width
	- Bools are treated as UInts
	- For other types of the same class are are the same: clone of any of the elements
	- Otherwise: fail
  

.. chisel:attr:: object chiselTypeOf

	Returns the chisel type of a hardware object, allowing other hardware to be constructed from it.  

.. chisel:attr:: object Input

		Input, Output, and Flipped are used to define the directions of Module IOs.
	
	Note that they currently clone their source argument, including its bindings.
	
	Thus, an error will be thrown if these are used on bound Data


.. chisel:attr:: object Output


.. chisel:attr:: object Flipped


.. chisel:attr:: abstract class Data extends HasId with NamedComponent with SourceInfoDoc

	This forms the root of the type system for wire data types. The data value	must be representable as some number (need not be known at Chisel compile
	time) of bits, and must have methods to pack / unpack structured data to /
	from bits.
	
	@groupdesc Connect Utilities for connecting hardware components
	@define coll data
  

	.. chisel:attr:: private[chisel3] def _assignCompatibilityExplicitDirection: Unit =

	
		This overwrites a relative SpecifiedDirection with an explicit one, and is used to implement	the compatibility layer where, at the elements, Flip is Input and unspecified is Output.
		DO NOT USE OUTSIDE THIS PURPOSE. THIS OPERATION IS DANGEROUS!
	    


	.. chisel:attr:: private[chisel3] def bind(target: Binding, parentDirection: SpecifiedDirection = SpecifiedDirection.Unspecified)

	
		Binds this node to the hardware graph.	parentDirection is the direction of the parent node, or Unspecified (default) if the target
		node is the top-level.
		binding and direction are valid after this call completes.
	    


	.. chisel:attr:: private[chisel3] def typeEquivalent(that: Data): Boolean

	
		Whether this Data has the same model ("data type") as that Data.	Data subtypes should overload this with checks against their own type.
	    


	.. chisel:attr:: def cloneType: this.type

	
		Internal API; Chisel users should look at chisel3.chiselTypeOf(...).	
		cloneType must be defined for any Chisel object extending Data.
		It is responsible for constructing a basic copy of the object being cloned.
		
		:return: a copy of the object.
		    


	.. chisel:attr:: private[chisel3] def cloneTypeFull: this.type =

	
		Internal API; Chisel users should look at chisel3.chiselTypeOf(...).	
		Returns a copy of this data type, with hardware bindings (if any) removed.
		Directionality data is still preserved.
	    


	.. chisel:attr:: final def := (that: Data)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): Unit

	
		Connect this $coll to that $coll mono-directionally and element-wise.	
		This uses the :chisel:reref:`MonoConnect`  algorithm.
		
		
		:param that: the $coll to connect to
			@group Connect
		    


	.. chisel:attr:: final def <> (that: Data)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): Unit

	
		Connect this $coll to that $coll bi-directionally and element-wise.	
		This uses the :chisel:reref:`BiConnect`  algorithm.
		
		
		:param that: the $coll to connect to
			@group Connect
		    


	.. chisel:attr:: def litOption(): Option[BigInt]

	
			If this is a literal that is representable as bits, returns the value as a BigInt.
		If not a literal, or not representable as bits (for example, is or contains Analog), returns None.
	   


	.. chisel:attr:: def litValue(): BigInt

	
			Returns the literal value if this is a literal that is representable as bits, otherwise crashes.
	   


	.. chisel:attr:: final def getWidth: Int

	
		Returns the width, in bits, if currently known. 
	


	.. chisel:attr:: final def isWidthKnown: Boolean

	
		Returns whether the width is currently known. 
	


	.. chisel:attr:: final def widthOption: Option[Int]

	
		Returns Some(width) if the width is known, else None. 
	


	.. chisel:attr:: def asTypeOf[T <: Data](that: T): T

	
		Does a reinterpret cast of the bits in this node into the format that provides.	Returns a new Wire of that type. Does not modify existing nodes.
		
		x.asTypeOf(that) performs the inverse operation of x := that.toBits.
		
		
		:note: bit widths are NOT checked, may pad or drop bits from input
		
		:note: that should have known widths
	    


	.. chisel:attr:: def do_asTypeOf[T <: Data](that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: private[chisel3] def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit

	
		Assigns this node from Bits type. Internal implementation for asTypeOf.    


	.. chisel:attr:: final def asUInt(): UInt

	
		Reinterpret cast to UInt.	
		
		:note: value not guaranteed to be preserved: for example, a SInt of width
			3 and value -1 (0b111) would become an UInt with value 7
		
		:note: Aggregates are recursively packed with the first element appearing
			in the least-significant bits of the result.
	    


	.. chisel:attr:: def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def toPrintable: Printable

	
		Default pretty printing 
	


.. chisel:attr:: trait WireFactory


	.. chisel:attr:: def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Construct a :chisel:reref:`Wire`  from a type template	
		:param t: The template from which to construct this wire
		    


.. chisel:attr:: object Wire extends WireFactory

	Utility for constructing hardware wires	
	The width of a `Wire` (inferred or not) is copied from the type template
	
	.. code-block:: scala 

		 val w0 = Wire(UInt()) // width is inferred
		 val w1 = Wire(UInt(8.W)) // width is set to 8
		
		 val w2 = Wire(Vec(4, UInt())) // width is inferred
		 val w3 = Wire(Vec(4, UInt(8.W))) // width of each element is set to 8
		
		 class MyBundle {
		   val unknown = UInt()
		   val known   = UInt(8.W)
		 }
		 val w4 = Wire(new MyBundle)
		 // Width of w4.unknown is inferred
		 // Width of w4.known is set to 8
	
	
  

.. chisel:attr:: object WireDefault

	Utility for constructing hardware wires with a default connection	
	The two forms of `WireDefault` differ in how the type and width of the resulting :chisel:reref:`Wire`  are
	specified.
	
	==Single Argument==
	The single argument form uses the argument to specify both the type and default connection. For
	non-literal :chisel:reref:`Bits` , the width of the :chisel:reref:`Wire`  will be inferred. For literal :chisel:reref:`Bits`  and all
	non-Bits arguments, the type will be copied from the argument. See the following examples for
	more details:
	
	1. Literal :chisel:reref:`Bits`  initializer: width will be set to match
	
	.. code-block:: scala 

		 val w1 = WireDefault(1.U) // width will be inferred to be 1
		 val w2 = WireDefault(1.U(8.W)) // width is set to 8
	
	
	2. Non-Literal :chisel:reref:`Element`  initializer - width will be inferred
	
	.. code-block:: scala 

		 val x = Wire(UInt())
		 val y = Wire(UInt(8.W))
		 val w1 = WireDefault(x) // width will be inferred
		 val w2 = WireDefault(y) // width will be inferred
	
	
	3. :chisel:reref:`Aggregate`  initializer - width will be set to match the aggregate
	
	
	.. code-block:: scala 

		 class MyBundle {
		   val unknown = UInt()
		   val known   = UInt(8.W)
		 }
		 val w1 = Wire(new MyBundle)
		 val w2 = WireDefault(w1)
		 // Width of w2.unknown is inferred
		 // Width of w2.known is set to 8
	
	
	==Double Argument==
	The double argument form allows the type of the :chisel:reref:`Wire`  and the default connection to be
	specified independently.
	
	The width inference semantics for `WireDefault` with two arguments match those of :chisel:reref:`Wire` . The
	first argument to `WireDefault` is the type template which defines the width of the `Wire` in
	exactly the same way as the only argument to :chisel:reref:`Wire` .
	
	More explicitly, you can reason about `WireDefault` with multiple arguments as if it were defined
	as:
	
	.. code-block:: scala 

		 def WireDefault[T <: Data](t: T, init: T): T = {
		   val x = Wire(t)
		   x := init
		   x
		 }
	
	
	
	:note: The `Default` in `WireDefault` refers to a `default` connection. This is in contrast to
		:chisel:reref:`RegInit`  where the `Init` refers to a value on reset.
  

	.. chisel:attr:: def apply[T <: Data](t: T, init: DontCare.type)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Construct a :chisel:reref:`Wire`  with a type template and a :chisel:reref:`chisel3.DontCare`  default	
		:param t: The type template used to construct this :chisel:reref:`Wire`
		
		:param init: The default connection to this :chisel:reref:`Wire` , can only be :chisel:reref:`DontCare`
		
		:note: This is really just a specialized form of `apply[T <: Data](t: T, init: T): T` with :chisel:reref:`DontCare`  as `init`
	    


	.. chisel:attr:: def apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Construct a :chisel:reref:`Wire`  with a type template and a default connection	
		:param t: The type template used to construct this :chisel:reref:`Wire`
		
		:param init: The hardware value that will serve as the default value
		    


	.. chisel:attr:: def apply[T <: Data](init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Construct a :chisel:reref:`Wire`  with a default connection	
		:param init: The hardware value that will serve as a type template and default value
		    


.. chisel:attr:: private[chisel3] object InternalDontCare extends Element

	RHS (source) for Invalidate API.	Causes connection logic to emit a DefInvalid when connected to an output port (or wire).
    

Assert.scala
------------
.. chisel:attr:: object assert


	.. chisel:attr:: def apply(cond: Bool, message: String, data: Bits*)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit

	
		Checks for a condition to be valid in the circuit at all times. If the	condition evaluates to false, the circuit simulation stops with an error.
		
		Does not fire when in reset (defined as the encapsulating Module's
		reset). If your definition of reset is not the encapsulating Module's
		reset, you will need to gate this externally.
		
		May be called outside of a Module (like defined in a function), so
		functions using assert make the standard Module assumptions (single clock
		and single reset).
		
		
		:param cond: condition, assertion fires (simulation fails) when false
		
		:param message: optional format string to print when the assertion fires
		
		:param data: optional bits to print in the message formatting
			
		
		:note: See :chisel:reref:`printf.apply(fmt:String* printf`  for format string documentation
		
		:note: currently cannot be used in core Chisel / libraries because macro
			defs need to be compiled first and the SBT project is not set up to do
			that
	    


	.. chisel:attr:: def apply(cond: Boolean, message: => String)

	
		An elaboration-time assertion, otherwise the same as the above run-time    * assertion. 


	.. chisel:attr:: def apply(cond: Boolean)

	
		A workaround for default-value overloading problems in Scala, just    * 'assert(cond, "")' 


.. chisel:attr:: object stop


	.. chisel:attr:: def apply(code: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit =

	
		Terminate execution with a failure code. 
	


	.. chisel:attr:: def apply()(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit =

	
		Terminate execution, indicating success. 
	


Reg.scala
---------
.. chisel:attr:: object Reg

	Utility for constructing hardware registers	
	The width of a `Reg` (inferred or not) is copied from the type template
	
	.. code-block:: scala 

		 val r0 = Reg(UInt()) // width is inferred
		 val r1 = Reg(UInt(8.W)) // width is set to 8
		
		 val r2 = Reg(Vec(4, UInt())) // width is inferred
		 val r3 = Reg(Vec(4, UInt(8.W))) // width of each element is set to 8
		
		 class MyBundle {
		   val unknown = UInt()
		   val known   = UInt(8.W)
		 }
		 val r4 = Reg(new MyBundle)
		 // Width of r4.unknown is inferred
		 // Width of r4.known is set to 8
	
	
  

	.. chisel:attr:: def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Construct a :chisel:reref:`Reg`  from a type template with no initialization value (reset is ignored).	Value will not change unless the :chisel:reref:`Reg`  is given a connection.
		
		:param t: The template from which to construct this wire
		    


.. chisel:attr:: object RegNext


	.. chisel:attr:: def apply[T <: Data](next: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Returns a register with the specified next and no reset initialization.	
		Essentially a 1-cycle delayed version of the input signal.
	    


	.. chisel:attr:: def apply[T <: Data](next: T, init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Returns a register with the specified next and reset initialization.	
		Essentially a 1-cycle delayed version of the input signal.
	    


.. chisel:attr:: object RegInit

	Utility for constructing hardware registers with an initialization value.	
	The register is set to the initialization value when the current implicit `reset` is high
	
	The two forms of `RegInit` differ in how the type and width of the resulting :chisel:reref:`Reg`  are
	specified.
	
	==Single Argument==
	The single argument form uses the argument to specify both the type and reset value. For
	non-literal :chisel:reref:`Bits` , the width of the :chisel:reref:`Reg`  will be inferred. For literal :chisel:reref:`Bits`  and all
	non-Bits arguments, the type will be copied from the argument. See the following examples for
	more details:
	
	1. Literal :chisel:reref:`Bits`  initializer: width will be set to match
	
	.. code-block:: scala 

		 val r1 = RegInit(1.U) // width will be inferred to be 1
		 val r2 = RegInit(1.U(8.W)) // width is set to 8
	
	
	2. Non-Literal :chisel:reref:`Element`  initializer - width will be inferred
	
	.. code-block:: scala 

		 val x = Wire(UInt())
		 val y = Wire(UInt(8.W))
		 val r1 = RegInit(x) // width will be inferred
		 val r2 = RegInit(y) // width will be inferred
	
	
	3. :chisel:reref:`Aggregate`  initializer - width will be set to match the aggregate
	
	
	.. code-block:: scala 

		 class MyBundle {
		   val unknown = UInt()
		   val known   = UInt(8.W)
		 }
		 val w1 = Reg(new MyBundle)
		 val w2 = RegInit(w1)
		 // Width of w2.unknown is inferred
		 // Width of w2.known is set to 8
	
	
	==Double Argument==
	The double argument form allows the type of the :chisel:reref:`Reg`  and the default connection to be
	specified independently.
	
	The width inference semantics for `RegInit` with two arguments match those of :chisel:reref:`Reg` . The
	first argument to `RegInit` is the type template which defines the width of the `Reg` in
	exactly the same way as the only argument to :chisel:reref:`Wire` .
	
	More explicitly, you can reason about `RegInit` with multiple arguments as if it were defined
	as:
	
	.. code-block:: scala 

		 def RegInit[T <: Data](t: T, init: T): T = {
		   val x = Reg(t)
		   x := init
		   x
		 }
	
  

	.. chisel:attr:: def apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Construct a :chisel:reref:`Reg`  from a type template initialized to the specified value on reset	
		:param t: The type template used to construct this :chisel:reref:`Reg`
		
		:param init: The value the :chisel:reref:`Reg`  is initialized to on reset
		    


	.. chisel:attr:: def apply[T <: Data](init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Construct a :chisel:reref:`Reg`  initialized on reset to the specified value.	
		:param init: Initial value that serves as a type template and reset value
		    


CompileOptions.scala
--------------------
.. chisel:attr:: trait CompileOptions


.. chisel:attr:: object CompileOptions


.. chisel:attr:: object ExplicitCompileOptions


Aggregate.scala
---------------
.. chisel:attr:: class AliasedAggregateFieldException(message: String) extends ChiselException(message)


.. chisel:attr:: sealed abstract class Aggregate extends Data

	An abstract class for data types that solely consist of (are an aggregate	of) other Data objects.
  

	.. chisel:attr:: def getElements: Seq[Data]

	
		Returns a Seq of the immediate contents of this Aggregate, in order.    


.. chisel:attr:: trait VecFactory extends SourceInfoDoc


	.. chisel:attr:: def apply[T <: Data](n: Int, gen: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =

	
		Creates a new :chisel:reref:`Vec`  with `n` entries of the specified data type.	
		
		:note: elements are NOT assigned by default and have no value
	    


	.. chisel:attr:: private[chisel3] def truncateIndex(idx: UInt, n: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =

	
		Truncate an index to implement modulo-power-of-2 addressing. 
	


.. chisel:attr:: sealed class Vec[T <: Data] private[chisel3] (gen: => T, val length: Int) extends Aggregate with VecLike[T]

	A vector (array) of :chisel:reref:`Data`  elements. Provides hardware versions of various	collection transformation functions found in software array implementations.
	
	Careful consideration should be given over the use of :chisel:reref:`Vec`  vs
	:chisel:reref:`scala.collection.immutable.Seq Seq`  or some other Scala collection. In general :chisel:reref:`Vec`  only
	needs to be used when there is a need to express the hardware collection in a :chisel:reref:`Reg`  or IO
	:chisel:reref:`Bundle`  or when access to elements of the array is indexed via a hardware signal.
	
	Example of indexing into a :chisel:reref:`Vec`  using a hardware address and where the :chisel:reref:`Vec`  is defined in
	an IO :chisel:reref:`Bundle`
	
	
	.. code-block:: scala 

		    val io = IO(new Bundle {
		      val in = Input(Vec(20, UInt(16.W)))
		      val addr = UInt(5.W)
		      val out = Output(UInt(16.W))
		    })
		    io.out := io.in(io.addr)
	
	
	
	:type-param T: type of elements
		
	
	:note:
		- when multiple conflicting assignments are performed on a Vec element, the last one takes effect (unlike Mem, where the result is undefined)
		- Vecs, unlike classes in Scala's collection library, are propagated intact to FIRRTL as a vector type, which may make debugging easier
  

	.. chisel:attr:: def <> (that: Seq[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit =

	
		Strong bulk connect, assigning elements in this Vec from elements in a Seq.	
		
		:note: the length of this Vec must match the length of the input Seq
	    


	.. chisel:attr:: def := (that: Seq[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit =

	
		Strong bulk connect, assigning elements in this Vec from elements in a Seq.	
		
		:note: the length of this Vec must match the length of the input Seq
	    


	.. chisel:attr:: override def apply(p: UInt): T

	
		Creates a dynamically indexed read or write accessor into the array.    


	.. chisel:attr:: def do_apply(p: UInt)(implicit compileOptions: CompileOptions): T =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def apply(idx: Int): T

	
		Creates a statically indexed read or write accessor into the array.    


	.. chisel:attr:: def toPrintable: Printable =

	
		Default "pretty-print" implementation	Analogous to printing a Seq
		Results in "Vec(elt0, elt1, ...)"
	    


.. chisel:attr:: object VecInit extends SourceInfoDoc


	.. chisel:attr:: def apply[T <: Data](elts: Seq[T]): Vec[T]

	
		Creates a new :chisel:reref:`Vec`  composed of elements of the input Seq of :chisel:reref:`Data` 	nodes.
		
		
		:note: input elements should be of the same type (this is checked at the
			FIRRTL level, but not at the Scala / Chisel level)
		
		:note: the width of all output elements is the width of the largest input
			element
		
		:note: output elements are connected from the input elements
	    


	.. chisel:attr:: def do_apply[T <: Data](elts: Seq[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def apply[T <: Data](elt0: T, elts: T*): Vec[T]

	
		Creates a new :chisel:reref:`Vec`  composed of the input :chisel:reref:`Data`  nodes.	
		
		:note: input elements should be of the same type (this is checked at the
			FIRRTL level, but not at the Scala / Chisel level)
		
		:note: the width of all output elements is the width of the largest input
			element
		
		:note: output elements are connected from the input elements
	    


	.. chisel:attr:: def do_apply[T <: Data](elt0: T, elts: T*)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T]

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T]

	
		Creates a new :chisel:reref:`Vec`  of length `n` composed of the results of the given	function applied over a range of integer values starting from 0.
		
		
		:param n: number of elements in the vector (the function is applied from
			0 to `n-1`)
		
		:param gen: function that takes in an Int (the index) and returns a
			:chisel:reref:`Data`  that becomes the output element
		    


	.. chisel:attr:: def do_tabulate[T <: Data](n: Int)(gen: (Int) => T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T]

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: trait VecLike[T <: Data] extends collection.IndexedSeq[T] with HasId with SourceInfoDoc

	A trait for :chisel:reref:`Vec` s containing common hardware generators for collection	operations.
  

	.. chisel:attr:: def do_apply(p: UInt)(implicit compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def forall(p: T => Bool): Bool

	
		Outputs true if p outputs true for every element.    


	.. chisel:attr:: def do_forall(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def exists(p: T => Bool): Bool

	
		Outputs true if p outputs true for at least one element.    


	.. chisel:attr:: def do_exists(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def contains(x: T)(implicit ev: T <:< UInt): Bool

	
		Outputs true if the vector contains at least one element equal to x (using	the === operator).
	    


	.. chisel:attr:: def do_contains(x: T)(implicit sourceInfo: SourceInfo, ev: T <:< UInt, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def count(p: T => Bool): UInt

	
		Outputs the number of elements for which p is true.    


	.. chisel:attr:: def do_count(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: private def indexWhereHelper(p: T => Bool)

	
		Helper function that appends an index (literal value) to each element,	useful for hardware generators which output an index.
	    


	.. chisel:attr:: def indexWhere(p: T => Bool): UInt

	
		Outputs the index of the first element for which p outputs true.    


	.. chisel:attr:: def do_indexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def lastIndexWhere(p: T => Bool): UInt

	
		Outputs the index of the last element for which p outputs true.    


	.. chisel:attr:: def do_lastIndexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def onlyIndexWhere(p: T => Bool): UInt

	
		Outputs the index of the element for which p outputs true, assuming that	the there is exactly one such element.
		
		The implementation may be more efficient than a priority mux, but
		incorrect results are possible if there is not exactly one true element.
		
		
		:note: the assumption that there is only one element for which p outputs
			true is NOT checked (useful in cases where the condition doesn't always
			hold, but the results are not used in those cases)
	    


	.. chisel:attr:: def do_onlyIndexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: abstract class Record(private[chisel3] implicit val compileOptions: CompileOptions) extends Aggregate

	Base class for Aggregates based on key values pairs of String and Data	
	Record should only be extended by libraries and fairly sophisticated generators.
	RTL writers should use :chisel:reref:`Bundle` .  See :chisel:reref:`Record#elements`  for an example.
  

	.. chisel:attr:: private[chisel3] def _makeLit(elems: (this.type => (Data, Data))*): this.type =

	
		Creates a Bundle literal of this type with specified values. this must be a chisel type.	
		
		:param elems: literal values, specified as a pair of the Bundle field to the literal value.
			The Bundle field is specified as a function from an object of this type to the field.
			Fields that aren't initialized to DontCare, and assignment to a wire will overwrite any
			existing value with DontCare.
		:return: a Bundle literal of this type with subelement values specified
			
		
		.. code-block:: scala 
	
			 class MyBundle extends Bundle {
			   val a = UInt(8.W)
			   val b = Bool()
			 }
			
			 (mew MyBundle).Lit(
			   _.a -> 42.U,
			   _.b -> true.B
			 )
		
	    


	.. chisel:attr:: override def toString: String =

	
		The collection of :chisel:reref:`Data` 	
		This underlying datastructure is a ListMap because the elements must
		remain ordered for serialization/deserialization. Elements added later
		are higher order when serialized (this is similar to :chisel:reref:`Vec` ). For example:
		
		.. code-block:: scala 
	
			   // Assume we have some type MyRecord that creates a Record from the ListMap
			   val record = MyRecord(ListMap("fizz" -> UInt(16.W), "buzz" -> UInt(16.W)))
			   // "buzz" is higher order because it was added later than "fizz"
			   record("fizz") := "hdead".U
			   record("buzz") := "hbeef".U
			   val uint = record.asUInt
			   assert(uint === "hbeefdead".U) // This will pass
		
	    


	.. chisel:attr:: def className: String

	
		Name for Pretty Printing 
	


	.. chisel:attr:: def toPrintable: Printable

	
		Default "pretty-print" implementation	Analogous to printing a Map
		Results in "`\$className(elt0.name -> elt0.value, ...)`"
	    


.. chisel:attr:: trait IgnoreSeqInBundle

		Mix-in for Bundles that have arbitrary Seqs of Chisel types that aren't
	involved in hardware construction.
	
	Used to avoid raising an error/exception when a Seq is a public member of the
	bundle.
	This is useful if we those public Seq fields in the Bundle are unrelated to
	hardware construction.
  

.. chisel:attr:: class AutoClonetypeException(message: String) extends ChiselException(message)  package experimental


.. chisel:attr:: abstract class Bundle(implicit compileOptions: CompileOptions) extends Record

	Base class for data types defined as a bundle of other data types.	
	Usage: extend this class (either as an anonymous or named class) and define
	members variables of :chisel:reref:`Data`  subtypes to be elements in the Bundle.
	
	Example of an anonymous IO bundle
	
	.. code-block:: scala 

		   class MyModule extends Module {
		     val io = IO(new Bundle {
		       val in = Input(UInt(64.W))
		       val out = Output(SInt(128.W))
		     })
		   }
	
	
	Or as a named class
	
	.. code-block:: scala 

		   class Packet extends Bundle {
		     val header = UInt(16.W)
		     val addr   = UInt(16.W)
		     val data   = UInt(32.W)
		   }
		   class MyModule extends Module {
		      val io = IO(new Bundle {
		        val inPacket = Input(new Packet)
		        val outPacket = Output(new Packet)
		      })
		      val reg = Reg(new Packet)
		      reg <> inPacket
		      outPacket <> reg
		   }
	
  

	.. chisel:attr:: def ignoreSeq: Boolean

	
			Overridden by :chisel:reref:`IgnoreSeqInBundle`  to allow arbitrary Seqs of Chisel elements.
	    


	.. chisel:attr:: private def getBundleField(m: java.lang.reflect.Method): Option[Data]

	
		Returns a field's contained user-defined Bundle element if it appears to	be one, otherwise returns None.
	    


	.. chisel:attr:: override def toPrintable: Printable

	
		Default "pretty-print" implementation	Analogous to printing a Map
		Results in "`Bundle(elt0.name -> elt0.value, ...)`"
		
		:note: The order is reversed from the order of elements in order to print
			the fields in the order they were defined
	    


MultiClock.scala
----------------
.. chisel:attr:: object withClockAndReset


	.. chisel:attr:: def apply[T](clock: Clock, reset: Reset)(block: => T): T =

	
		Creates a new Clock and Reset scope	
		
		:param clock: the new implicit Clock
		
		:param reset: the new implicit Reset
		
		:param block: the block of code to run with new implicit Clock and Reset
		:return: the result of the block
		    


.. chisel:attr:: object withClock


	.. chisel:attr:: def apply[T](clock: Clock)(block: => T): T =

	
		Creates a new Clock scope	
		
		:param clock: the new implicit Clock
		
		:param block: the block of code to run with new implicit Clock
		:return: the result of the block
		    


.. chisel:attr:: object withReset


	.. chisel:attr:: def apply[T](reset: Reset)(block: => T): T =

	
		Creates a new Reset scope	
		
		:param reset: the new implicit Reset
		
		:param block: the block of code to run with new implicit Reset
		:return: the result of the block
		    


Annotation.scala
----------------
.. chisel:attr:: trait ChiselAnnotation

	Interface for Annotations in Chisel	
	Defines a conversion to a corresponding FIRRTL Annotation
  

	.. chisel:attr:: def toFirrtl: Annotation

	
		Conversion to FIRRTL Annotation 
	


.. chisel:attr:: trait RunFirrtlTransform extends ChiselAnnotation

	Mixin for :chisel:reref:`ChiselAnnotation`  that instantiates an associated FIRRTL Transform when this Annotation is present	during a run of
	:chisel:reref:`Driver$.execute(args:Array[String],dut:()=>chisel3\.experimental\.RawModule)* Driver.execute` .
	Automatic Transform instantiation is *not* supported when the Circuit and Annotations are serialized before invoking
	FIRRTL.
  

.. chisel:attr:: final case class ChiselLegacyAnnotation private[chisel3] (component: InstanceId, transformClass: Class[_ <: Transform], value: String) extends ChiselAnnotation with RunFirrtlTransform


.. chisel:attr:: private[chisel3] object ChiselLegacyAnnotation  object annotate


.. chisel:attr:: object annotate


.. chisel:attr:: object dontTouch

	Marks that a signal should not be removed by Chisel and Firrtl optimization passes	
	
	.. code-block:: scala 

		 class MyModule extends Module {
		   val io = IO(new Bundle {
		     val a = Input(UInt(32.W))
		     val b = Output(UInt(32.W))
		   })
		   io.b := io.a
		   val dead = io.a +% 1.U // normally dead would be pruned by DCE
		   dontTouch(dead) // Marking it as such will preserve it
		 }
	
	
	
	:note: Calling this on :chisel:reref:`Data`  creates an annotation that Chisel emits to a separate annotations
		file. This file must be passed to FIRRTL independently of the `.fir` file. The execute methods
		in :chisel:reref:`chisel3.Driver`  will pass the annotations to FIRRTL automatically.
  

	.. chisel:attr:: def apply[T <: Data](data: T)(implicit compileOptions: CompileOptions): T =

	
		Marks a signal to be preserved in Chisel and Firrtl	
		
		:note: Requires the argument to be bound to hardware
		
		:param data: The signal to be marked
		:return: Unmodified signal `data`
		    


.. chisel:attr:: object doNotDedup

	Marks that a module to be ignored in Dedup Transform in Firrtl pass	
	
	.. code-block:: scala 

		  def fullAdder(a: UInt, b: UInt, myName: String): UInt = {
		    val m = Module(new Module {
		      val io = IO(new Bundle {
		        val a = Input(UInt(32.W))
		        val b = Input(UInt(32.W))
		        val out = Output(UInt(32.W))
		      })
		      override def desiredName = "adder_" + myNname
		      io.out := io.a + io.b
		    })
		    doNotDedup(m)
		    m.io.a := a
		    m.io.b := b
		    m.io.out
		  }
		
		class AdderTester extends Module
		  with ConstantPropagationTest {
		  val io = IO(new Bundle {
		    val a = Input(UInt(32.W))
		    val b = Input(UInt(32.W))
		    val out = Output(Vec(2, UInt(32.W)))
		  })
		
		  io.out(0) := fullAdder(io.a, io.b, "mod1")
		  io.out(1) := fullAdder(io.a, io.b, "mod2")
		 }
	
	
	
	:note: Calling this on :chisel:reref:`Data`  creates an annotation that Chisel emits to a separate annotations
		file. This file must be passed to FIRRTL independently of the `.fir` file. The execute methods
		in :chisel:reref:`chisel3.Driver`  will pass the annotations to FIRRTL automatically.
  

	.. chisel:attr:: def apply[T <: LegacyModule](module: T)(implicit compileOptions: CompileOptions): Unit =

	
		Marks a module to be ignored in Dedup Transform in Firrtl	
		
		:param data: The module to be marked
		:return: Unmodified signal `module`
		    


RawModule.scala
---------------
.. chisel:attr:: abstract class RawModule(implicit moduleCompileOptions: CompileOptions) extends BaseModule

	Abstract base class for Modules that contain Chisel RTL.	This abstract base class is a user-defined module which does not include implicit clock and reset and supports
	multiple IO() declarations.
  

.. chisel:attr:: abstract class MultiIOModule(implicit moduleCompileOptions: CompileOptions) extends RawModule

	Abstract base class for Modules, which behave much like Verilog modules.	These may contain both logic and state which are written in the Module
	body (constructor).
	This abstract base class includes an implicit clock and reset.
	
	
	:note: Module instantiations must be wrapped in a Module() call.
  

.. chisel:attr:: abstract class LegacyModule(implicit moduleCompileOptions: CompileOptions) extends MultiIOModule

	Legacy Module class that restricts IOs to just io, clock, and reset, and provides a constructor	for threading through explicit clock and reset.
	
	While this class isn't planned to be removed anytime soon (there are benefits to restricting
	IO), the clock and reset constructors will be phased out. Recommendation is to wrap the module
	in a withClock/withReset/withClockAndReset block, or directly hook up clock or reset IO pins.
  

Attach.scala
------------
.. chisel:attr:: object attach


	.. chisel:attr:: def apply(elts: Analog*)(implicit sourceInfo: SourceInfo): Unit =

	
		Create an electrical connection between :chisel:reref:`Analog`  components	
		
		:param elts: The components to attach
			
			@example
		
		.. code-block:: scala 
	
			 val a1 = Wire(Analog(32.W))
			 val a2 = Wire(Analog(32.W))
			 attach(a1, a2)
		
	    


SeqUtils.scala
--------------
.. chisel:attr:: private[chisel3] object SeqUtils


	.. chisel:attr:: def asUInt[T <: Bits](in: Seq[T]): UInt

	
		Concatenates the data elements of the input sequence, in sequence order, together.	The first element of the sequence forms the least significant bits, while the last element
		in the sequence forms the most significant bits.
		
		Equivalent to r(n-1) ## ... ## r(1) ## r(0).
	    


	.. chisel:attr:: def do_asUInt[T <: Bits](in: Seq[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =

	
		@group SourceInfoTransformMacros 
	


	.. chisel:attr:: def count(in: Seq[Bool]): UInt

	
		Outputs the number of elements that === true.B.    


	.. chisel:attr:: def do_count(in: Seq[Bool])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacros 
	


	.. chisel:attr:: def priorityMux[T <: Data](in: Seq[(Bool, T)]): T

	
		Returns the data value corresponding to the first true predicate.    


	.. chisel:attr:: def do_priorityMux[T <: Data](in: Seq[(Bool, T)]) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		@group SourceInfoTransformMacros 
	


	.. chisel:attr:: def oneHotMux[T <: Data](in: Iterable[(Bool, T)]): T

	
		Returns the data value corresponding to the lone true predicate.	This is elaborated to firrtl using a structure that should be optimized into and and/or tree.
		
		
		:note: assumes exactly one true predicate, results undefined otherwise
			FixedPoint values or aggregates containing FixedPoint values cause this optimized structure to be lost
	    


	.. chisel:attr:: def do_oneHotMux[T <: Data](in: Iterable[(Bool, T)]) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		@group SourceInfoTransformMacros 
	


package.scala
-------------
.. chisel:attr:: package object chisel3

	This package contains the main chisel3 API. 

.. chisel:attr:: implicit class fromBigIntToLiteral(bigint: BigInt)

		These implicit classes allow one to convert scala.Int|scala.BigInt to
	Chisel.UInt|Chisel.SInt by calling .asUInt|.asSInt on them, respectively.
	The versions .asUInt(width)|.asSInt(width) are also available to explicitly
	mark a width for the new literal.
	
	Also provides .asBool to scala.Boolean and .asUInt to String
	
	Note that, for stylistic reasons, one should avoid extracting immediately
	after this call using apply, ie. 0.asUInt(1)(0) due to potential for
	confusion (the 1 is a bit length and the 0 is a bit extraction position).
	Prefer storing the result and then extracting from it.
	
	Implementation note: the empty parameter list (like `U()`) is necessary to prevent
	interpreting calls that have a non-Width parameter as a chained apply, otherwise things like
	`0.asUInt(16)` (instead of `16.W`) compile without error and produce undesired results.
        

	.. chisel:attr:: def B: Bool

	
		Int to Bool conversion, allowing compact syntax like 1.B and 0.B          


	.. chisel:attr:: def U: UInt

	
		Int to UInt conversion, recommended style for constants.          


	.. chisel:attr:: def S: SInt

	
		Int to SInt conversion, recommended style for constants.          


	.. chisel:attr:: def U(width: Width): UInt

	
		Int to UInt conversion with specified width, recommended style for constants.          


	.. chisel:attr:: def S(width: Width): SInt

	
		Int to SInt conversion with specified width, recommended style for constants.          


	.. chisel:attr:: def asUInt(): UInt

	
		Int to UInt conversion, recommended style for variables.          


	.. chisel:attr:: def asSInt(): SInt

	
		Int to SInt conversion, recommended style for variables.          


	.. chisel:attr:: def asUInt(width: Width): UInt

	
		Int to UInt conversion with specified width, recommended style for variables.          


	.. chisel:attr:: def asSInt(width: Width): SInt

	
		Int to SInt conversion with specified width, recommended style for variables.          


	.. chisel:attr:: def U: UInt

	
		String to UInt parse, recommended style for constants.          


	.. chisel:attr:: def U(width: Width): UInt

	
		String to UInt parse with specified width, recommended style for constants.          


	.. chisel:attr:: def asUInt(): UInt =

	
		String to UInt parse, recommended style for variables.          


	.. chisel:attr:: def asUInt(width: Width): UInt

	
		String to UInt parse with specified width, recommended style for variables.          


	.. chisel:attr:: def B: Bool

	
		Boolean to Bool conversion, recommended style for constants.          


	.. chisel:attr:: def asBool(): Bool

	
		Boolean to Bool conversion, recommended style for variables.          


.. chisel:attr:: implicit class PrintableHelper(val sc: StringContext) extends AnyVal

	Implicit for custom Printable string interpolator 


	.. chisel:attr:: def p(args: Any*): Printable =

	
		Custom string interpolator for generating Printables: p"..."	Will call .toString on any non-Printable arguments (mimicking s"...")
	      


.. chisel:attr:: case class ExpectedChiselTypeException(message: String) extends BindingException(message)

	A function expected a Chisel type but got a hardware object    

.. chisel:attr:: case class ExpectedHardwareException(message: String) extends BindingException(message)

	A function expected a hardware object but got a Chisel type    

.. chisel:attr:: case class MixedDirectionAggregateException(message: String) extends BindingException(message)

	An aggregate had a mix of specified and unspecified directionality children    

.. chisel:attr:: case class RebindingException(message: String) extends BindingException(message)

	Attempted to re-bind an already bound (directionality or hardware) object    

StrongEnum.scala
----------------
.. chisel:attr:: object EnumAnnotations


.. chisel:attr:: case class EnumComponentAnnotation(target: Named, enumTypeName: String) extends SingleTargetAnnotation[Named]

	An annotation for strong enum instances that are ''not'' inside of Vecs	
	
	:param target: the enum instance being annotated
	
	:param typeName: the name of the enum's type (e.g. ''"mypackage.MyEnum"'')
	    

.. chisel:attr:: case class EnumVecAnnotation(target: Named, typeName: String, fields: Seq[Seq[String]]) extends SingleTargetAnnotation[Named]

	An annotation for Vecs of strong enums.	
	The ''fields'' parameter deserves special attention, since it may be difficult to understand. Suppose you create a the following Vec:
	
	.. code-block:: scala 

		               VecInit(new Bundle {
		                 val e = MyEnum()
		                 val b = new Bundle {
		                   val inner_e = MyEnum()
		                 }
		                 val v = Vec(3, MyEnum())
		               }
	
	
	Then, the ''fields'' parameter will be: ''Seq(Seq("e"), Seq("b", "inner_e"), Seq("v"))''. Note that for any Vec that doesn't contain Bundles, this field will simply be an empty Seq.
	
	
	:param target: the Vec being annotated
	
	:param typeName: the name of the enum's type (e.g. ''"mypackage.MyEnum"'')
	
	:param fields: a list of all chains of elements leading from the Vec instance to its inner enum fields.
		
	    

.. chisel:attr:: case class EnumDefAnnotation(typeName: String, definition: Map[String, BigInt]) extends NoTargetAnnotation

	An annotation for enum types (rather than enum ''instances'').	
	
	:param typeName: the name of the enum's type (e.g. ''"mypackage.MyEnum"'')
	
	:param definition: a map describing which integer values correspond to which enum names
	    

.. chisel:attr:: abstract class EnumType(private val factory: EnumFactory, selfAnnotating: Boolean


.. chisel:attr:: abstract class EnumFactory


.. chisel:attr:: private[chisel3] object EnumMacros


.. chisel:attr:: private[chisel3] class UnsafeEnum(override val width: Width) extends EnumType(UnsafeEnum, selfAnnotating


.. chisel:attr:: private object UnsafeEnum extends EnumFactory  


Printf.scala
------------
.. chisel:attr:: object printf

	Prints a message in simulation	
	See apply methods for use
  

	.. chisel:attr:: private[chisel3] def format(formatIn: String): String =

	
		Helper for packing escape characters 
	


	.. chisel:attr:: def apply(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit

	
		Prints a message in simulation	
		Prints a message every cycle. If defined within the scope of a :chisel:reref:`when`  block, the message
		will only be printed on cycles that the when condition is true.
		
		Does not fire when in reset (defined as the encapsulating Module's reset). If your definition
		of reset is not the encapsulating Module's reset, you will need to gate this externally.
		
		May be called outside of a Module (like defined in a function), uses the current default clock
		and reset. These can be overriden with :chisel:reref:`withClockAndReset` .
		
		==Format Strings==
		
		This method expects a ''format string'' and an ''argument list'' in a similar style to printf
		in C. The format string expects a :chisel:reref:`scala.Predef.String String`  that may contain ''format
		specifiers'' For example:
		
		.. code-block:: scala 
	
			   printf("myWire has the value %d\n", myWire)
		
		This prints the string "myWire has the value " followed by the current value of `myWire` (in
		decimal, followed by a newline.
		
		There must be exactly as many arguments as there are format specifiers
		
		===Format Specifiers===
		
		Format specifiers are prefixed by `%`. If you wish to print a literal `%`, use `%%`.
		- `%d` - Decimal
		- `%x` - Hexadecimal
		- `%b` - Binary
		- `%c` - 8-bit Character
		- `%n` - Name of a signal
		- `%N` - Full name of a leaf signal (in an aggregate)
		
		
		:param fmt: printf format string
		
		:param data: format string varargs containing data to print
		    


	.. chisel:attr:: def apply(pable: Printable)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit =

	
		Prints a message in simulation	
		Prints a message every cycle. If defined within the scope of a :chisel:reref:`when`  block, the message
		will only be printed on cycles that the when condition is true.
		
		Does not fire when in reset (defined as the encapsulating Module's reset). If your definition
		of reset is not the encapsulating Module's reset, you will need to gate this externally.
		
		May be called outside of a Module (like defined in a function), uses the current default clock
		and reset. These can be overriden with :chisel:reref:`withClockAndReset` .
		
		@see :chisel:reref:`Printable`  documentation
		
		:param pable: :chisel:reref:`Printable`  to print
		    


Mem.scala
---------
.. chisel:attr:: object Mem


	.. chisel:attr:: def apply[T <: Data](size: BigInt, t: T): Mem[T]

	
		Creates a combinational/asynchronous-read, sequential/synchronous-write :chisel:reref:`Mem` .	
		
		:param size: number of elements in the memory
		
		:param t: data type of memory element
		    


	.. chisel:attr:: def apply[T <: Data](size: Int, t: T): Mem[T]

	
		Creates a combinational/asynchronous-read, sequential/synchronous-write :chisel:reref:`Mem` .	
		
		:param size: number of elements in the memory
		
		:param t: data type of memory element
		    


	.. chisel:attr:: def do_apply[T <: Data](size: BigInt, t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Mem[T] =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_apply[T <: Data](size: Int, t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Mem[T]

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: sealed abstract class MemBase[T <: Data](t: T, val length: BigInt) extends HasId with NamedComponent with SourceInfoDoc


	.. chisel:attr:: def apply(x: BigInt): T

	
		Creates a read accessor into the memory with static addressing. See the	class documentation of the memory for more detailed information.
	    


	.. chisel:attr:: def apply(x: Int): T

	
		Creates a read accessor into the memory with static addressing. See the	class documentation of the memory for more detailed information.
	    


	.. chisel:attr:: def do_apply(idx: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_apply(idx: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def apply(x: UInt): T

	
		Creates a read/write accessor into the memory with dynamic addressing.	See the class documentation of the memory for more detailed information.
	    


	.. chisel:attr:: def do_apply(idx: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def read(x: UInt): T

	
		Creates a read accessor into the memory with dynamic addressing. See the	class documentation of the memory for more detailed information.
	    


	.. chisel:attr:: def do_read(idx: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def write(idx: UInt, data: T)(implicit compileOptions: CompileOptions): Unit =

	
		Creates a write accessor into the memory.	
		
		:param idx: memory element index to write into
		
		:param data: new data to write
		    


	.. chisel:attr:: def write(idx: UInt, data: T, mask: Seq[Bool]) (implicit evidence: T <:< Vec[_], compileOptions: CompileOptions): Unit =

	
		Creates a masked write accessor into the memory.	
		
		:param idx: memory element index to write into
		
		:param data: new data to write
		
		:param mask: write mask as a Seq of Bool: a write to the Vec element in
			memory is only performed if the corresponding mask index is true.
			
		
		:note: this is only allowed if the memory's element data type is a Vec
	    


.. chisel:attr:: sealed class Mem[T <: Data] private (t: T, length: BigInt) extends MemBase(t, length)

	A combinational/asynchronous-read, sequential/synchronous-write memory.	
	Writes take effect on the rising clock edge after the request. Reads are
	combinational (requests will return data on the same cycle).
	Read-after-write hazards are not an issue.
	
	
	:note: when multiple conflicting writes are performed on a Mem element, the
		result is undefined (unlike Vec, where the last assignment wins)
  

.. chisel:attr:: sealed class Mem[T <: Data] private (t: T, length: BigInt) extends MemBase(t, length)  object SyncReadMem


.. chisel:attr:: object SyncReadMem


	.. chisel:attr:: def apply[T <: Data](size: BigInt, t: T): SyncReadMem[T]

	
		Creates a sequential/synchronous-read, sequential/synchronous-write :chisel:reref:`SyncReadMem` .	
		
		:param size: number of elements in the memory
		
		:param t: data type of memory element
		    


	.. chisel:attr:: def apply[T <: Data](size: Int, t: T): SyncReadMem[T]

	
		Creates a sequential/synchronous-read, sequential/synchronous-write :chisel:reref:`SyncReadMem` .	
		
		:param size: number of elements in the memory
		
		:param t: data type of memory element
		    


	.. chisel:attr:: def do_apply[T <: Data](size: BigInt, t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SyncReadMem[T] =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_apply[T <: Data](size: Int, t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SyncReadMem[T]

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: sealed class SyncReadMem[T <: Data] private (t: T, n: BigInt) extends MemBase[T](t, n)

	A sequential/synchronous-read, sequential/synchronous-write memory.	
	Writes take effect on the rising clock edge after the request. Reads return
	data on the rising edge after the request. Read-after-write behavior (when
	a read and write to the same address are requested on the same cycle) is
	undefined.
	
	
	:note: when multiple conflicting writes are performed on a Mem element, the
		result is undefined (unlike Vec, where the last assignment wins)
  

	.. chisel:attr:: def do_read(addr: UInt, enable: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		@group SourceInfoTransformMacro 
	


Bits.scala
----------
.. chisel:attr:: abstract class Element extends Data

	Element is a leaf data type: it cannot contain other :chisel:reref:`Data`  objects. Example uses are for representing primitive	data types, like integers and bits.
	
	@define coll element
  

.. chisel:attr:: private[chisel3] sealed trait ToBoolable extends Element

	Exists to unify common interfaces of :chisel:reref:`Bits`  and :chisel:reref:`Reset` .	
	
	:note: This is a workaround because macros cannot override abstract methods.
  

	.. chisel:attr:: final def asBool(): Bool

	
		Casts this $coll to a :chisel:reref:`Bool` 	
		
		:note: The width must be known and equal to 1
	    


	.. chisel:attr:: def do_asBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def toBool(): Bool

	
		Casts this $coll to a :chisel:reref:`Bool` 	
		
		:note: The width must be known and equal to 1
	    


	.. chisel:attr:: def do_toBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: sealed abstract class Bits(private[chisel3] val width: Width) extends Element with ToBoolable

	A data type for values represented by a single bitvector. This provides basic bitwise operations.	
	@groupdesc Bitwise Bitwise hardware operators
	@define coll :chisel:reref:`Bits`
	
	@define sumWidthInt    :note: The width of the returned $coll is `width of this` + `that`.
	
	@define sumWidth       :note: The width of the returned $coll is `width of this` + `width of that`.
	
	@define unchangedWidth :note: The width of the returned $coll is unchanged, i.e., the `width of this`.
  

	.. chisel:attr:: final def tail(n: Int): UInt

	
		Tail operator	
		
		:param n: the number of bits to remove
		:return: This $coll with the `n` most significant bits removed.
			@group Bitwise
		    


	.. chisel:attr:: final def head(n: Int): UInt

	
		Head operator	
		
		:param n: the number of bits to take
		:return: The `n` most significant bits of this $coll
			@group Bitwise
		    


	.. chisel:attr:: def do_tail(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_head(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def apply(x: BigInt): Bool

	
		Returns the specified bit on this $coll as a :chisel:reref:`Bool` , statically addressed.	
		
		:param x: an index
		:return: the specified bit
		    


	.. chisel:attr:: final def do_apply(x: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def apply(x: Int): Bool

	
		Returns the specified bit on this $coll as a :chisel:reref:`Bool` , statically addressed.	
		
		:param x: an index
		:return: the specified bit
		
		:note: convenience method allowing direct use of :chisel:reref:`scala.Int`  without implicits
	    


	.. chisel:attr:: final def do_apply(x: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def apply(x: UInt): Bool

	
		Returns the specified bit on this wire as a :chisel:reref:`Bool` , dynamically addressed.	
		
		:param x: a hardware component whose value will be used for dynamic addressing
		:return: the specified bit
		    


	.. chisel:attr:: final def do_apply(x: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def apply(x: Int, y: Int): UInt

	
		Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.	
		@example
		
		.. code-block:: scala 
	
			 myBits = 0x5 = 0b101
			 myBits(1,0) => 0b01  // extracts the two least significant bits
		
		
		:param x: the high bit
		
		:param y: the low bit
		:return: a hardware component contain the requested bits
		    


	.. chisel:attr:: final def do_apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def apply(x: BigInt, y: BigInt): UInt

	
		Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.	
		@example
		
		.. code-block:: scala 
	
			 myBits = 0x5 = 0b101
			 myBits(1,0) => 0b01  // extracts the two least significant bits
		
		
		:param x: the high bit
		
		:param y: the low bit
		:return: a hardware component contain the requested bits
		    


	.. chisel:attr:: final def do_apply(x: BigInt, y: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def pad(that: Int): this.type

	
		Pad operator	
		
		:param that: the width to pad to
		:return: this @coll zero padded up to width `that`. If `that` is less than the width of the original component,
			this method returns the original component.
		
		:note: For :chisel:reref:`SInt` s only, this will do sign extension.
			@group Bitwise
	    


	.. chisel:attr:: def do_pad(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def unary_~ (): Bits

	
		Bitwise inversion operator	
		:return: this $coll with each bit inverted
			@group Bitwise
		    


	.. chisel:attr:: def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def << (that: BigInt): Bits

	
		Static left shift operator	
		
		:param that: an amount to shift by
		:return: this $coll with `that` many zeros concatenated to its least significant end
			$sumWidthInt
			@group Bitwise
		    


	.. chisel:attr:: def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def << (that: Int): Bits

	
		Static left shift operator	
		
		:param that: an amount to shift by
		:return: this $coll with `that` many zeros concatenated to its least significant end
			$sumWidthInt
			@group Bitwise
		    


	.. chisel:attr:: def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def << (that: UInt): Bits

	
		Dynamic left shift operator	
		
		:param that: a hardware component
		:return: this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
		
		:note: The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
			@group Bitwise
	    


	.. chisel:attr:: def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def >> (that: BigInt): Bits

	
		Static right shift operator	
		
		:param that: an amount to shift by
		:return: this $coll with `that` many least significant bits truncated
			$unchangedWidth
			@group Bitwise
		    


	.. chisel:attr:: def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def >> (that: Int): Bits

	
		Static right shift operator	
		
		:param that: an amount to shift by
		:return: this $coll with `that` many least significant bits truncated
			$unchangedWidth
			@group Bitwise
		    


	.. chisel:attr:: def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def >> (that: UInt): Bits

	
		Dynamic right shift operator	
		
		:param that: a hardware component
		:return: this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
			significant bits.
			$unchangedWidth
			@group Bitwise
		    


	.. chisel:attr:: def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def toBools(): Seq[Bool]

	
		Returns the contents of this wire as a :chisel:reref:`scala.collection.Seq`  of :chisel:reref:`Bool` . 
	


	.. chisel:attr:: def do_toBools(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Seq[Bool]

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def asBools(): Seq[Bool]

	
		Returns the contents of this wire as a :chisel:reref:`scala.collection.Seq`  of :chisel:reref:`Bool` . 
	


	.. chisel:attr:: def do_asBools(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Seq[Bool]

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def asSInt(): SInt

	
		Reinterpret this $coll as an :chisel:reref:`SInt` 	
		
		:note: The arithmetic value is not preserved if the most-significant bit is set. For example, a :chisel:reref:`UInt`  of
			width 3 and value 7 (0b111) would become an :chisel:reref:`SInt`  of width 3 and value -1.
	    


	.. chisel:attr:: def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def asFixedPoint(that: BinaryPoint): FixedPoint

	
		Reinterpret this $coll as a :chisel:reref:`FixedPoint` .	
		
		:note: The value is not guaranteed to be preserved. For example, a :chisel:reref:`UInt`  of width 3 and value 7 (0b111) would
			become a :chisel:reref:`FixedPoint`  with value -1. The interpretation of the number is also affected by the specified binary
			point. '''Caution is advised!'''
	    


	.. chisel:attr:: def do_asFixedPoint(that: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def ## (that: Bits): UInt

	
		Concatenation operator	
		
		:param that: a hardware component
		:return: this $coll concatenated to the most significant end of `that`
			$sumWidth
			@group Bitwise
		    


	.. chisel:attr:: def do_## (that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def toPrintable: Printable

	
		Default print as :chisel:reref:`Decimal`  
	


.. chisel:attr:: abstract trait Num[T <: Data]

	Abstract trait defining operations available on numeric-like hardware data types.	
	
	:type-param T: the underlying type of the number
		@groupdesc Arithmetic Arithmetic hardware operators
		@groupdesc Comparison Comparison hardware operators
		@groupdesc Logical Logical hardware operators
		@define coll numeric-like type
		@define numType hardware type
		@define canHaveHighCost can result in significant cycle time and area costs
		@define canGenerateA This method generates a
	
	@define singleCycleMul  :note: $canGenerateA fully combinational multiplier which $canHaveHighCost.
	
	@define singleCycleDiv  :note: $canGenerateA fully combinational divider which $canHaveHighCost.
	
	@define maxWidth        :note: The width of the returned $numType is `max(width of this, width of that)`.
	
	@define maxWidthPlusOne :note: The width of the returned $numType is `max(width of this, width of that) + 1`.
	
	@define sumWidth        :note: The width of the returned $numType is `width of this` + `width of that`.
	
	@define unchangedWidth  :note: The width of the returned $numType is unchanged, i.e., the `width of this`.
  

	.. chisel:attr:: final def + (that: T): T

	
		Addition operator	
		
		:param that: a $numType
		:return: the sum of this $coll and `that`
			$maxWidth
			@group Arithmetic
		    


	.. chisel:attr:: def do_+ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def * (that: T): T

	
		Multiplication operator	
		
		:param that: a $numType
		:return: the product of this $coll and `that`
			$sumWidth
			$singleCycleMul
			@group Arithmetic
		    


	.. chisel:attr:: def do_* (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def / (that: T): T

	
		Division operator	
		
		:param that: a $numType
		:return: the quotient of this $coll divided by `that`
			$singleCycleDiv
			@todo full rules
			@group Arithmetic
		    


	.. chisel:attr:: def do_/ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def % (that: T): T

	
		Modulo operator	
		
		:param that: a $numType
		:return: the remainder of this $coll divided by `that`
			$singleCycleDiv
			@group Arithmetic
		    


	.. chisel:attr:: def do_% (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def - (that: T): T

	
		Subtraction operator	
		
		:param that: a $numType
		:return: the difference of this $coll less `that`
			$maxWidthPlusOne
			@group Arithmetic
		    


	.. chisel:attr:: def do_- (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def < (that: T): Bool

	
		Less than operator	
		
		:param that: a $numType
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is less than `that`
			@group Comparison
		    


	.. chisel:attr:: def do_< (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def <= (that: T): Bool

	
		Less than or equal to operator	
		
		:param that: a $numType
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is less than or equal to `that`
			@group Comparison
		    


	.. chisel:attr:: def do_<= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def > (that: T): Bool

	
		Greater than operator	
		
		:param that: a hardware component
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is greater than `that`
			@group Comparison
		    


	.. chisel:attr:: def do_> (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def >= (that: T): Bool

	
		Greater than or equal to operator	
		
		:param that: a hardware component
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is greather than or equal to `that`
			@group Comparison
		    


	.. chisel:attr:: def do_>= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def abs(): T

	
		Absolute value operator	
		:return: a $numType with a value equal to the absolute value of this $coll
			$unchangedWidth
			@group Arithmetic
		    


	.. chisel:attr:: def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def min(that: T): T

	
		Minimum operator	
		
		:param that: a hardware $coll
		:return: a $numType with a value equal to the mimimum value of this $coll and `that`
			$maxWidth
			@group Arithmetic
		    


	.. chisel:attr:: def do_min(that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def max(that: T): T

	
		Maximum operator	
		
		:param that: a $numType
		:return: a $numType with a value equal to the mimimum value of this $coll and `that`
			$maxWidth
			@group Arithmetic
		    


	.. chisel:attr:: def do_max(that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: sealed class UInt private[chisel3] (width: Width) extends Bits(width) with Num[UInt]

	A data type for unsigned integers, represented as a binary bitvector. Defines arithmetic operations between other	integer types.
	
	@define coll :chisel:reref:`UInt`
	@define numType $coll
	
	@define expandingWidth :note: The width of the returned $coll is `width of this` + `1`.
	
	@define constantWidth  :note: The width of the returned $coll is unchanged, i.e., `width of this`.
  

	.. chisel:attr:: final def unary_- (): UInt

	
		Unary negation (expanding width)	
		:return: a $coll equal to zero minus this $coll
			$constantWidth
			@group Arithmetic
		    


	.. chisel:attr:: final def unary_-% (): UInt

	
		Unary negation (constant width)	
		:return: a $coll equal to zero minus this $coll shifted right by one.
			$constantWidth
			@group Arithmetic
		    


	.. chisel:attr:: def do_unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def * (that: SInt): SInt

	
		Multiplication operator	
		
		:param that: a hardware :chisel:reref:`SInt`
		:return: the product of this $coll and `that`
			$sumWidth
			$singleCycleMul
			@group Arithmetic
		    


	.. chisel:attr:: def do_* (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def +& (that: UInt): UInt

	
		Addition operator (expanding width)	
		
		:param that: a hardware $coll
		:return: the sum of this $coll and `that`
			$maxWidthPlusOne
			@group Arithmetic
		    


	.. chisel:attr:: final def +% (that: UInt): UInt

	
		Addition operator (constant width)	
		
		:param that: a hardware $coll
		:return: the sum of this $coll and `that`
			$maxWidth
			@group Arithmetic
		    


	.. chisel:attr:: final def -& (that: UInt): UInt

	
		Subtraction operator (increasing width)	
		
		:param that: a hardware $coll
		:return: the difference of this $coll less `that`
			$maxWidthPlusOne
			@group Arithmetic
		    


	.. chisel:attr:: final def -% (that: UInt): UInt

	
		Subtraction operator (constant width)	
		
		:param that: a hardware $coll
		:return: the difference of this $coll less `that`
			$maxWidth
			@group Arithmetic
		    


	.. chisel:attr:: def do_+& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_+% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_-& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_-% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def & (that: UInt): UInt

	
		Bitwise and operator	
		
		:param that: a hardware $coll
		:return: the bitwise and of  this $coll and `that`
			$maxWidth
			@group Bitwise
		    


	.. chisel:attr:: final def | (that: UInt): UInt

	
		Bitwise or operator	
		
		:param that: a hardware $coll
		:return: the bitwise or of this $coll and `that`
			$maxWidth
			@group Bitwise
		    


	.. chisel:attr:: final def ^ (that: UInt): UInt

	
		Bitwise exclusive or (xor) operator	
		
		:param that: a hardware $coll
		:return: the bitwise xor of this $coll and `that`
			$maxWidth
			@group Bitwise
		    


	.. chisel:attr:: def do_& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_| (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_^ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def orR(): Bool

	
		Or reduction operator	
		:return: a hardware :chisel:reref:`Bool`  resulting from every bit of this $coll or'd together
			@group Bitwise
		    


	.. chisel:attr:: final def andR(): Bool

	
		And reduction operator	
		:return: a hardware :chisel:reref:`Bool`  resulting from every bit of this $coll and'd together
			@group Bitwise
		    


	.. chisel:attr:: final def xorR(): Bool

	
		Exclusive or (xor) reduction operator	
		:return: a hardware :chisel:reref:`Bool`  resulting from every bit of this $coll xor'd together
			@group Bitwise
		    


	.. chisel:attr:: def do_orR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_andR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_xorR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def =/= (that: UInt): Bool

	
		Dynamic not equals operator	
		
		:param that: a hardware $coll
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is not equal to `that`
			@group Comparison
		    


	.. chisel:attr:: final def === (that: UInt): Bool

	
		Dynamic equals operator	
		
		:param that: a hardware $coll
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is equal to `that`
			@group Comparison
		    


	.. chisel:attr:: def do_=/= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_=== (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def unary_! () : Bool

	
		Unary not	
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll equals zero
			@group Bitwise
		    


	.. chisel:attr:: def do_unary_! (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def bitSet(off: UInt, dat: Bool): UInt

	
		Conditionally set or clear a bit	
		
		:param off: a dynamic offset
		
		:param dat: set if true, clear if false
		:return: a hrdware $coll with bit `off` set or cleared based on the value of `dat`
			$unchangedWidth
		    


	.. chisel:attr:: def do_bitSet(off: UInt, dat: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def zext(): SInt

	
		Zero extend as :chisel:reref:`SInt` 	
		:return: an :chisel:reref:`SInt`  equal to this $coll with an additional zero in its most significant bit
		
		:note: The width of the returned :chisel:reref:`SInt`  is `width of this` + `1`.
	    


	.. chisel:attr:: def do_zext(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: trait UIntFactory


	.. chisel:attr:: def apply(): UInt

	
		Create a UInt type with inferred width. 
	


	.. chisel:attr:: def apply(width: Width): UInt

	
		Create a UInt port with specified width. 
	


	.. chisel:attr:: protected[chisel3] def Lit(value: BigInt, width: Width): UInt =

	
		Create a UInt literal with specified width. 
	


	.. chisel:attr:: def apply(range: Range): UInt =

	
		Create a UInt with the specified range 
	


	.. chisel:attr:: def apply(range: (NumericBound[Int], NumericBound[Int])): UInt =

	
		Create a UInt with the specified range 
	


.. chisel:attr:: sealed class SInt private[chisel3] (width: Width) extends Bits(width) with Num[SInt]

	A data type for signed integers, represented as a binary bitvector. Defines arithmetic operations between other	integer types.
	
	@define coll :chisel:reref:`SInt`
	@define numType $coll
	
	@define expandingWidth :note: The width of the returned $coll is `width of this` + `1`.
	
	@define constantWidth  :note: The width of the returned $coll is unchanged, i.e., `width of this`.
  

	.. chisel:attr:: final def unary_- (): SInt

	
		Unary negation (expanding width)	
		:return: a hardware $coll equal to zero minus this $coll
			$constantWidth
			@group Arithmetic
		    


	.. chisel:attr:: final def unary_-% (): SInt

	
		Unary negation (constant width)	
		:return: a hardware $coll equal to zero minus `this` shifted right by one
			$constantWidth
			@group Arithmetic
		    


	.. chisel:attr:: def unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: override def do_+ (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		add (default - no growth) operator 
	


	.. chisel:attr:: override def do_- (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		subtract (default - no growth) operator 
	


	.. chisel:attr:: final def * (that: UInt): SInt

	
		Multiplication operator	
		
		:param that: a hardware $coll
		:return: the product of this $coll and `that`
			$sumWidth
			$singleCycleMul
			@group Arithmetic
		    


	.. chisel:attr:: def do_* (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def +& (that: SInt): SInt

	
		Addition operator (expanding width)	
		
		:param that: a hardware $coll
		:return: the sum of this $coll and `that`
			$maxWidthPlusOne
			@group Arithmetic
		    


	.. chisel:attr:: final def +% (that: SInt): SInt

	
		Addition operator (constant width)	
		
		:param that: a hardware $coll
		:return: the sum of this $coll and `that` shifted right by one
			$maxWidth
			@group Arithmetic
		    


	.. chisel:attr:: final def -& (that: SInt): SInt

	
		Subtraction operator (increasing width)	
		
		:param that: a hardware $coll
		:return: the difference of this $coll less `that`
			$maxWidthPlusOne
			@group Arithmetic
		    


	.. chisel:attr:: final def -% (that: SInt): SInt

	
		Subtraction operator (constant width)	
		
		:param that: a hardware $coll
		:return: the difference of this $coll less `that` shifted right by one
			$maxWidth
			@group Arithmetic
		    


	.. chisel:attr:: def do_+& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_+% (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_-& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_-% (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def & (that: SInt): SInt

	
		Bitwise and operator	
		
		:param that: a hardware $coll
		:return: the bitwise and of  this $coll and `that`
			$maxWidth
			@group Bitwise
		    


	.. chisel:attr:: final def | (that: SInt): SInt

	
		Bitwise or operator	
		
		:param that: a hardware $coll
		:return: the bitwise or of this $coll and `that`
			$maxWidth
			@group Bitwise
		    


	.. chisel:attr:: final def ^ (that: SInt): SInt

	
		Bitwise exclusive or (xor) operator	
		
		:param that: a hardware $coll
		:return: the bitwise xor of this $coll and `that`
			$maxWidth
			@group Bitwise
		    


	.. chisel:attr:: def do_& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_| (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_^ (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def =/= (that: SInt): Bool

	
		Dynamic not equals operator	
		
		:param that: a hardware $coll
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is not equal to `that`
			@group Comparison
		    


	.. chisel:attr:: final def === (that: SInt): Bool

	
		Dynamic equals operator	
		
		:param that: a hardware $coll
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is equal to `that`
			@group Comparison
		    


	.. chisel:attr:: def do_=/= (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_=== (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: trait SIntFactory


	.. chisel:attr:: def apply(): SInt

	
		Create an SInt type with inferred width. 
	


	.. chisel:attr:: def apply(width: Width): SInt

	
		Create a SInt type or port with fixed width. 
	


	.. chisel:attr:: def apply(range: Range): SInt =

	
		Create a SInt with the specified range 
	


	.. chisel:attr:: def apply(range: (NumericBound[Int], NumericBound[Int])): SInt =

	
		Create a SInt with the specified range 
	


	.. chisel:attr:: protected[chisel3] def Lit(value: BigInt, width: Width): SInt =

	
		Create an SInt literal with specified width. 
	


.. chisel:attr:: object SInt extends SIntFactory  sealed trait Reset extends Element with ToBoolable


.. chisel:attr:: sealed trait Reset extends Element with ToBoolable


.. chisel:attr:: sealed class Bool() extends UInt(1.W) with Reset

	A data type for booleans, defined as a single bit indicating true or false.	
	@define coll :chisel:reref:`Bool`
	@define numType $coll
  

	.. chisel:attr:: def litToBooleanOption: Option[Boolean]

	
		Convert to a :chisel:reref:`scala.Option`  of :chisel:reref:`scala.Boolean`  
	


	.. chisel:attr:: def litToBoolean: Boolean

	
		Convert to a :chisel:reref:`scala.Boolean`  
	


	.. chisel:attr:: final def & (that: Bool): Bool

	
		Bitwise and operator	
		
		:param that: a hardware $coll
		:return: the bitwise and of  this $coll and `that`
			@group Bitwise
		    


	.. chisel:attr:: final def | (that: Bool): Bool

	
		Bitwise or operator	
		
		:param that: a hardware $coll
		:return: the bitwise or of this $coll and `that`
			@group Bitwise
		    


	.. chisel:attr:: final def ^ (that: Bool): Bool

	
		Bitwise exclusive or (xor) operator	
		
		:param that: a hardware $coll
		:return: the bitwise xor of this $coll and `that`
			@group Bitwise
		    


	.. chisel:attr:: def do_& (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_| (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_^ (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: override def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def || (that: Bool): Bool

	
		Logical or operator	
		
		:param that: a hardware $coll
		:return: the lgocial or of this $coll and `that`
		
		:note: this is equivalent to :chisel:reref:`Bool!.|(that:chisel3\.Bool)* Bool.|)`
			@group Logical
	    


	.. chisel:attr:: def do_|| (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def && (that: Bool): Bool

	
		Logical and operator	
		
		:param that: a hardware $coll
		:return: the lgocial and of this $coll and `that`
		
		:note: this is equivalent to :chisel:reref:`Bool!.&(that:chisel3\.Bool)* Bool.&`
			@group Logical
	    


	.. chisel:attr:: def do_&& (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def asClock(): Clock

	
		Reinterprets this $coll as a clock 
	


	.. chisel:attr:: def do_asClock(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Clock

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: trait BoolFactory


	.. chisel:attr:: def apply(): Bool

	
		Creates an empty Bool.   


	.. chisel:attr:: protected[chisel3] def Lit(x: Boolean): Bool =

	
		Creates Bool literal.   


.. chisel:attr:: object Bool extends BoolFactory  package experimental


.. chisel:attr:: sealed class FixedPoint private(width: Width, val binaryPoint: BinaryPoint) extends Bits(width) with Num[FixedPoint]

	A sealed class representing a fixed point number that has a bit width and a binary point The width and binary point	may be inferred.
	
	IMPORTANT: The API provided here is experimental and may change in the future.
	
	
	:param width:       bit width of the fixed point number
	
	:param binaryPoint: the position of the binary point with respect to the right most bit of the width currently this
		should be positive but it is hoped to soon support negative points and thus use this field as a
		simple exponent
		@define coll           :chisel:reref:`FixedPoint`
		@define numType        $coll
	
	@define expandingWidth :note: The width of the returned $coll is `width of this` + `1`.
	
	@define constantWidth  :note: The width of the returned $coll is unchanged, i.e., `width of this`.
    

	.. chisel:attr:: def litToDoubleOption: Option[Double]

	
		Convert to a :chisel:reref:`scala.Option`  of :chisel:reref:`scala.Boolean`  
	


	.. chisel:attr:: def litToDouble: Double

	
		Convert to a :chisel:reref:`scala.Option`  
	


	.. chisel:attr:: final def unary_- (): FixedPoint

	
		Unary negation (expanding width)	
		:return: a hardware $coll equal to zero minus this $coll
			$expandingWidth
			@group Arithmetic
		      


	.. chisel:attr:: final def unary_-% (): FixedPoint

	
		Unary negation (constant width)	
		:return: a hardware $coll equal to zero minus `this` shifted right by one
			$constantWidth
			@group Arithmetic
		      


	.. chisel:attr:: def unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: override def do_+ (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		add (default - no growth) operator 
	


	.. chisel:attr:: override def do_- (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		subtract (default - no growth) operator 
	


	.. chisel:attr:: final def * (that: UInt): FixedPoint

	
		Multiplication operator	
		
		:param that: a hardware :chisel:reref:`UInt`
		:return: the product of this $coll and `that`
			$sumWidth
			$singleCycleMul
			@group Arithmetic
		      


	.. chisel:attr:: def do_* (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def * (that: SInt): FixedPoint

	
		Multiplication operator	
		
		:param that: a hardware :chisel:reref:`SInt`
		:return: the product of this $coll and `that`
			$sumWidth
			$singleCycleMul
			@group Arithmetic
		      


	.. chisel:attr:: def do_* (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def +& (that: FixedPoint): FixedPoint

	
		Addition operator (expanding width)	
		
		:param that: a hardware $coll
		:return: the sum of this $coll and `that`
			$maxWidthPlusOne
			@group Arithmetic
		      


	.. chisel:attr:: final def +% (that: FixedPoint): FixedPoint

	
		Addition operator (constant width)	
		
		:param that: a hardware $coll
		:return: the sum of this $coll and `that` shifted right by one
			$maxWidth
			@group Arithmetic
		      


	.. chisel:attr:: final def -& (that: FixedPoint): FixedPoint

	
		Subtraction operator (increasing width)	
		
		:param that: a hardware $coll
		:return: the difference of this $coll less `that`
			$maxWidthPlusOne
			@group Arithmetic
		      


	.. chisel:attr:: final def -% (that: FixedPoint): FixedPoint

	
		Subtraction operator (constant width)	
		
		:param that: a hardware $coll
		:return: the difference of this $coll less `that` shifted right by one
			$maxWidth
			@group Arithmetic
		      


	.. chisel:attr:: def do_+& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_+% (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_-& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_-% (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def & (that: FixedPoint): FixedPoint

	
		Bitwise and operator	
		
		:param that: a hardware $coll
		:return: the bitwise and of  this $coll and `that`
			$maxWidth
			@group Bitwise
		      


	.. chisel:attr:: final def | (that: FixedPoint): FixedPoint

	
		Bitwise or operator	
		
		:param that: a hardware $coll
		:return: the bitwise or of this $coll and `that`
			$maxWidth
			@group Bitwise
		      


	.. chisel:attr:: final def ^ (that: FixedPoint): FixedPoint

	
		Bitwise exclusive or (xor) operator	
		
		:param that: a hardware $coll
		:return: the bitwise xor of this $coll and `that`
			$maxWidth
			@group Bitwise
		      


	.. chisel:attr:: def do_& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_| (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_^ (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_setBinaryPoint(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: final def =/= (that: FixedPoint): Bool

	
		Dynamic not equals operator	
		
		:param that: a hardware $coll
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is not equal to `that`
			@group Comparison
		      


	.. chisel:attr:: final def === (that: FixedPoint): Bool

	
		Dynamic equals operator	
		
		:param that: a hardware $coll
		:return: a hardware :chisel:reref:`Bool`  asserted if this $coll is equal to `that`
			@group Comparison
		      


	.. chisel:attr:: def do_!= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_=/= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_=== (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: sealed trait PrivateType private case object PrivateObject extends PrivateType

	Use PrivateObject to force users to specify width and binaryPoint by name    

.. chisel:attr:: object FixedPoint

		Factory and convenience methods for the FixedPoint class
	IMPORTANT: The API provided here is experimental and may change in the future.
    

	.. chisel:attr:: def apply(): FixedPoint

	
		Create an FixedPoint type with inferred width. 
	


	.. chisel:attr:: def apply(width: Width, binaryPoint: BinaryPoint): FixedPoint

	
		Create an FixedPoint type or port with fixed width. 
	


	.. chisel:attr:: def fromBigInt(value: BigInt, width: Width, binaryPoint: BinaryPoint): FixedPoint =

	
		Create an FixedPoint literal with inferred width from BigInt.	Use PrivateObject to force users to specify width and binaryPoint by name
	      


	.. chisel:attr:: def fromBigInt(value: BigInt, binaryPoint: BinaryPoint = 0.BP): FixedPoint =

	
		Create an FixedPoint literal with inferred width from BigInt.	Use PrivateObject to force users to specify width and binaryPoint by name
	      


	.. chisel:attr:: def fromBigInt(value: BigInt, width: Int, binaryPoint: Int): FixedPoint

	
		Create an FixedPoint literal with inferred width from BigInt.	Use PrivateObject to force users to specify width and binaryPoint by name
	      


	.. chisel:attr:: def fromDouble(value: Double, width: Width, binaryPoint: BinaryPoint): FixedPoint =

	
		Create an FixedPoint literal with inferred width from Double.	Use PrivateObject to force users to specify width and binaryPoint by name
	      


	.. chisel:attr:: def apply(value: BigInt, width: Width, binaryPoint: BinaryPoint): FixedPoint =

	
		Create an FixedPoint port with specified width and binary position. 
	


	.. chisel:attr:: def toBigInt(x: Double, binaryPoint: Int): BigInt =

	
			How to create a bigint from a double with a specific binaryPoint
		
		:param x:           a double value
		
		:param binaryPoint: a binaryPoint that you would like to use
			@return
		      


	.. chisel:attr:: def toDouble(i: BigInt, binaryPoint: Int): Double =

	
			converts a bigInt with the given binaryPoint into the double representation
		
		:param i:           a bigint
		
		:param binaryPoint: the implied binaryPoint of @i
			@return
		      


.. chisel:attr:: final class Analog private (private[chisel3] val width: Width) extends Element

	Data type for representing bidirectional bitvectors of a given width	
	Analog support is limited to allowing wiring up of Verilog BlackBoxes with bidirectional (inout)
	pins. There is currently no support for reading or writing of Analog types within Chisel code.
	
	Given that Analog is bidirectional, it is illegal to assign a direction to any Analog type. It
	is legal to "flip" the direction (since Analog can be a member of aggregate types) which has no
	effect.
	
	Analog types are generally connected using the bidirectional :chisel:reref:`attach`  mechanism, but also
	support limited bulkconnect `<>`. Analog types are only allowed to be bulk connected *once* in a
	given module. This is to prevent any surprising consequences of last connect semantics.
	
	
	:note: This API is experimental and subject to change
    

.. chisel:attr:: object Analog

	Object that provides factory methods for :chisel:reref:`Analog`  objects	
	
	:note: This API is experimental and subject to change
    

Printable.scala
---------------
.. chisel:attr:: sealed abstract class Printable

	Superclass of things that can be printed in the resulting circuit	
	Usually created using the custom string interpolator `p"..."`. Printable string interpolation is
	similar to :chisel:reref:`https://docs.scala-lang.org/overviews/core/string-interpolation.html String
	interpolation in Scala`  For example:
	
	.. code-block:: scala 

		   printf(p"The value of wire = \$wire\n")
	
	This is equivalent to writing:
	
	.. code-block:: scala 

		   printf(p"The value of wire = %d\n", wire)
	
	All Chisel data types have a method `.toPrintable` that gives a default pretty print that can be
	accessed via `p"..."`. This works even for aggregate types, for example:
	
	.. code-block:: scala 

		   val myVec = VecInit(5.U, 10.U, 13.U)
		   printf(p"myVec = \$myVec\n")
		   // myVec = Vec(5, 10, 13)
		
		   val myBundle = Wire(new Bundle {
		     val foo = UInt()
		     val bar = UInt()
		   })
		   myBundle.foo := 3.U
		   myBundle.bar := 11.U
		   printf(p"myBundle = \$myBundle\n")
		   // myBundle = Bundle(a -> 3, b -> 11)
	
	Users can override the default behavior of `.toPrintable` in custom :chisel:reref:`Bundle`  and :chisel:reref:`Record`
	types.
  

	.. chisel:attr:: def unpack(ctx: Component): (String, Iterable[String])

	
		Unpack into format String and a List of String arguments (identifiers)	
		:note: This must be called after elaboration when Chisel nodes actually
			have names
	    


	.. chisel:attr:: final def +(that: Printable): Printables

	
		Allow for appending Printables like Strings 
	


	.. chisel:attr:: final def +(that: String): Printables

	
		Allow for appending Strings to Printables 
	


.. chisel:attr:: object Printable


	.. chisel:attr:: def pack(fmt: String, data: Data*): Printable =

	
		Pack standard printf fmt, args* style into Printable    


.. chisel:attr:: case class Printables(pables: Iterable[Printable]) extends Printable


.. chisel:attr:: case class PString(str: String) extends Printable

	Wrapper for printing Scala Strings 


.. chisel:attr:: sealed abstract class FirrtlFormat(private[chisel3] val specifier: Char) extends Printable

	Superclass for Firrtl format specifiers for Bits 


.. chisel:attr:: object FirrtlFormat


	.. chisel:attr:: def apply(specifier: String, data: Data): FirrtlFormat =

	
		Helper for constructing Firrtl Formats	Accepts data to simplify pack
	    


.. chisel:attr:: case class Decimal(bits: Bits) extends FirrtlFormat('d')

	Format bits as Decimal 


.. chisel:attr:: case class Hexadecimal(bits: Bits) extends FirrtlFormat('x')

	Format bits as Hexidecimal 


.. chisel:attr:: case class Binary(bits: Bits) extends FirrtlFormat('b')

	Format bits as Binary 


.. chisel:attr:: case class Character(bits: Bits) extends FirrtlFormat('c')

	Format bits as Character 


.. chisel:attr:: case class Name(data: Data) extends Printable

	Put innermost name (eg. field of bundle) 


.. chisel:attr:: case class FullName(data: Data) extends Printable

	Put full name within parent namespace (eg. bundleName.field) 


.. chisel:attr:: case object Percent extends Printable

	Represents escaped percents 


Clock.scala
-----------
.. chisel:attr:: object Clock


.. chisel:attr:: sealed class Clock(private[chisel3] val width: Width


	.. chisel:attr:: def toPrintable: Printable

	
		Not really supported 
	


BlackBox.scala
--------------
.. chisel:attr:: sealed abstract class Param case class IntParam(value: BigInt) extends Param

	Parameters for BlackBoxes 


.. chisel:attr:: case class RawParam(value: String) extends Param

	Unquoted String 


.. chisel:attr:: abstract class ExtModule(val params: Map[String, Param] = Map.empty[String, Param]) extends BaseBlackBox

	Defines a black box, which is a module that can be referenced from within	Chisel, but is not defined in the emitted Verilog. Useful for connecting
	to RTL modules defined outside Chisel.
	
	A variant of BlackBox, this has a more consistent naming scheme in allowing
	multiple top-level IO and does not drop the top prefix.
	
	@example
	Some design require a differential input clock to clock the all design.
	With the xilinx FPGA for example, a Verilog template named IBUFDS must be
	integrated to use differential input:
	
	.. code-block:: scala 

		  IBUFDS #(.DIFF_TERM("TRUE"),
		           .IOSTANDARD("DEFAULT")) ibufds (
		   .IB(ibufds_IB),
		   .I(ibufds_I),
		   .O(ibufds_O)
		  );
	
	
	To instantiate it, a BlackBox can be used like following:
	
	.. code-block:: scala 

		 import chisel3._
		 import chisel3.experimental._
		
		 // Example with Xilinx differential buffer IBUFDS
		 class IBUFDS extends ExtModule(Map("DIFF_TERM" -> "TRUE", // Verilog parameters
		                                    "IOSTANDARD" -> "DEFAULT"
		                      )) {
		   val O = IO(Output(Clock()))
		   val I = IO(Input(Clock()))
		   val IB = IO(Input(Clock()))
		 }
	
	
	:note: The parameters API is experimental and may change
    

.. chisel:attr:: abstract class BlackBox(val params: Map[String, Param] = Map.empty[String, Param])(implicit compileOptions: CompileOptions) extends BaseBlackBox

	Defines a black box, which is a module that can be referenced from within	Chisel, but is not defined in the emitted Verilog. Useful for connecting
	to RTL modules defined outside Chisel.
	
	@example
	Some design require a differential input clock to clock the all design.
	With the xilinx FPGA for example, a Verilog template named IBUFDS must be
	integrated to use differential input:
	
	.. code-block:: scala 

		  IBUFDS #(.DIFF_TERM("TRUE"),
		           .IOSTANDARD("DEFAULT")) ibufds (
		   .IB(ibufds_IB),
		   .I(ibufds_I),
		   .O(ibufds_O)
		  );
	
	
	To instantiate it, a BlackBox can be used like following:
	
	.. code-block:: scala 

		 import chisel3._
		 import chisel3.experimental._
		
		 // Example with Xilinx differential buffer IBUFDS
		 class IBUFDS extends BlackBox(Map("DIFF_TERM" -> "TRUE", // Verilog parameters
		                                   "IOSTANDARD" -> "DEFAULT"
		                      )) {
		   val io = IO(new Bundle {
		     val O = Output(Clock()) // IO names will be the same
		     val I = Input(Clock())  // (without 'io_' in prefix)
		     val IB = Input(Clock()) //
		   })
		 }
	
	
	:note: The parameters API is experimental and may change
  

.. chisel:attr:: abstract class BlackBox(val params: Map[String, Param]


Module.scala
------------
.. chisel:attr:: object Module extends SourceInfoDoc


	.. chisel:attr:: def apply[T <: BaseModule](bc: => T): T

	
		A wrapper method that all Module instantiations must be wrapped in	(necessary to help Chisel track internal state).
		
		
		:param bc: the Module being created
			
		:return: the input module `m` with Chisel metadata properly set
		    


	.. chisel:attr:: def do_apply[T <: BaseModule](bc: => T) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def clock: Clock

	
		Returns the implicit Clock 
	


	.. chisel:attr:: def reset: Reset

	
		Returns the implicit Reset 
	


	.. chisel:attr:: def currentModule: Option[BaseModule]

	
		Returns the current Module 
	


	.. chisel:attr:: def apply[T<:Data](iodef: T): T =

	
		Constructs a port for the current Module	
		This must wrap the datatype used to set the io field of any Module.
		i.e. All concrete modules must have defined io in this form:
		[lazy] val io[: io type] = IO(...[: io type])
		
		Items in [] are optional.
		
		The granted iodef must be a chisel type and not be bound to hardware.
		
		Also registers a Data as a port, also performing bindings. Cannot be called once ports are
		requested (so that all calls to ports will return the same information).
		Internal API.
	      


.. chisel:attr:: abstract class BaseModule extends HasId

	Abstract base class for Modules, an instantiable organizational unit for RTL.    

	.. chisel:attr:: private[chisel3] def isClosed

	
		Internal check if a Module is closed 
	


	.. chisel:attr:: private[chisel3] def generateComponent(): Component

	
		Generates the FIRRTL Component (Module or Blackbox) of this Module.	Also closes the module so no more construction can happen inside.
	      


	.. chisel:attr:: private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit

	
		Sets up this module in the parent context      


	.. chisel:attr:: def desiredName: String

	
		Desired name of this module. Override this to give this module a custom, perhaps parametric,	name.
	      


	.. chisel:attr:: final def toNamed: ModuleName

	
		Returns a FIRRTL ModuleName that references this object	
		:note: Should not be called until circuit elaboration is complete
	      


	.. chisel:attr:: private[chisel3] def getChiselPorts: Seq[(String, Data)] =

	
			Internal API. Returns a list of this module's generated top-level ports as a map of a String
		(FIRRTL name) to the IO object. Only valid after the module is closed.
		
		Note: for BlackBoxes (but not ExtModules), this returns the contents of the top-level io
		object, consistent with what is emitted in FIRRTL.
		
		TODO: Use SeqMap/VectorMap when those data structures become available.
	     


	.. chisel:attr:: protected def nameIds(rootClass: Class[_]): HashMap[HasId, String] =

	
		Called at the Module.apply(...) level after this Module has finished elaborating.	Returns a map of nodes -> names, for named nodes.
		
		Helper method.
	      


	.. chisel:attr:: def cleanName(name: String): String

	
		Scala generates names like chisel3$util$Queue$$ram for private vals	This extracts the part after $$ for names like this and leaves names
		without $$ unchanged
	        


	.. chisel:attr:: def _compatAutoWrapPorts()

	
		Compatibility function. Allows Chisel2 code which had ports without the IO wrapper to	compile under Bindings checks. Does nothing in non-compatibility mode.
		
		Should NOT be used elsewhere. This API will NOT last.
		
		TODO: remove this, perhaps by removing Bindings checks in compatibility mode.
	      


	.. chisel:attr:: protected def _bindIoInPlace(iodef: Data): Unit =

	
		Chisel2 code didn't require the IO(...) wrapper and would assign a Chisel type directly to	io, then do operations on it. This binds a Chisel type in-place (mutably) as an IO.
	      


	.. chisel:attr:: private[chisel3] def bindIoInPlace(iodef: Data): Unit

	
		Private accessor for _bindIoInPlace 
	


	.. chisel:attr:: protected def IO[T<:Data](iodef: T): T

	
			This must wrap the datatype used to set the io field of any Module.
		i.e. All concrete modules must have defined io in this form:
		[lazy] val io[: io type] = IO(...[: io type])
		
		Items in [] are optional.
		
		The granted iodef must be a chisel type and not be bound to hardware.
		
		Also registers a Data as a port, also performing bindings. Cannot be called once ports are
		requested (so that all calls to ports will return the same information).
		Internal API.
		
		TODO(twigg): Specifically walk the Data definition to call out which nodes
		are problematic.
	     


	.. chisel:attr:: override def instanceName: String

	
		Signal name (for simulation). 
	


