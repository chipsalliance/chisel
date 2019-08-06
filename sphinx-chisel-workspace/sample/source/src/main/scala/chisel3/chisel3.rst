----------------------
src/main/scala/chisel3
----------------------

.. toctree::
	testers/testers.rst
	internal/internal.rst
	util/util.rst
	stage/stage.rst


Driver.scala
------------
.. chisel:attr:: trait BackendCompilationUtilities extends FirrtlBackendCompilationUtilities

		The Driver provides methods to invoke the chisel3 compiler and the firrtl compiler.
	By default firrtl is automatically run after chisel.  an :chisel:reref:`ExecutionOptionsManager`
	is needed to manage options.  It can parser command line arguments or coordinate
	multiple chisel toolchain tools options.
	
	@example
	
	.. code-block:: scala 

		          val optionsManager = new ExecutionOptionsManager("chisel3")
		              with HasFirrtlOptions
		              with HasChiselExecutionOptions {
		            commonOptions = CommonOption(targetDirName = "my_target_dir")
		            chiselOptions = ChiselExecutionOptions(runFirrtlCompiler = false)
		          }
		          chisel3.Driver.execute(optionsManager, () => new Dut)
	
	or via command line arguments
	
	.. code-block:: scala 

		          args = "--no-run-firrtl --target-dir my-target-dir".split(" +")
		          chisel3.execute(args, () => new DUT)
	
  

	.. chisel:attr:: def compileFirrtlToVerilog(prefix: String, dir: File): Boolean =

	
		Compile Chirrtl to Verilog by invoking Firrtl inside the same JVM	
		
		:param prefix: basename of the file
		
		:param dir:    directory where file lives
		:return:       true if compiler completed successfully
		    


.. chisel:attr:: trait ChiselExecutionResult

		This family provides return values from the chisel3 and possibly firrtl compile steps
  

.. chisel:attr:: case class ChiselExecutionSuccess(circuitOption: Option[Circuit], emitted: String, firrtlResultOption: Option[FirrtlExecutionResult] ) extends ChiselExecutionResult

		
	
	:param circuitOption:  Optional circuit, has information like circuit name
	
	:param emitted:            The emitted Chirrrl text
	
	:param firrtlResultOption: Optional Firrtl result, @see freechipsproject/firrtl for details
	  

.. chisel:attr:: case class ChiselExecutionFailure(message: String) extends ChiselExecutionResult

		Getting one of these indicates failure of some sort.
	
	
	:param message: A clue might be provided here.
	  

.. chisel:attr:: case class ChiselExecutionFailure(message: String) extends ChiselExecutionResult  object Driver extends BackendCompilationUtilities


.. chisel:attr:: object Driver extends BackendCompilationUtilities


	.. chisel:attr:: def elaborate[T <: RawModule](gen: () => T): Circuit

	
			Elaborate the Module specified in the gen function into a Chisel IR Circuit.
		
		
		:param gen: A function that creates a Module hierarchy.
		:return: The resulting Chisel IR in the form of a Circuit. (TODO: Should be FIRRTL IR)
		    


	.. chisel:attr:: def toFirrtl(ir: Circuit): firrtl.ir.Circuit

	
			Convert the given Chisel IR Circuit to a FIRRTL Circuit.
		
		
		:param ir: Chisel IR Circuit, generated e.g. by elaborate().
		    


	.. chisel:attr:: def emit[T <: RawModule](gen: () => T): String

	
			Emit the Module specified in the gen function directly as a FIRRTL string without
		invoking FIRRTL.
		
		
		:param gen: A function that creates a Module hierarchy.
		    


	.. chisel:attr:: def emit[T <: RawModule](ir: Circuit): String

	
			Emit the given Chisel IR Circuit as a FIRRTL string, without invoking FIRRTL.
		
		
		:param ir: Chisel IR Circuit, generated e.g. by elaborate().
		    


	.. chisel:attr:: def emitVerilog[T <: RawModule](gen: => T): String =

	
			Elaborate the Module specified in the gen function into Verilog.
		
		
		:param gen: A function that creates a Module hierarchy.
		:return: A String containing the design in Verilog.
		    


	.. chisel:attr:: def dumpFirrtl(ir: Circuit, optName: Option[File]): File =

	
			Dump the elaborated Chisel IR Circuit as a FIRRTL String, without invoking FIRRTL.
		
		If no File is given as input, it will dump to a default filename based on the name of the
		top Module.
		
		
		:param c: Elaborated Chisel Circuit.
		
		:param optName: File to dump to. If unspecified, defaults to "<topmodule>.fir".
		:return: The File the circuit was dumped to.
		    


	.. chisel:attr:: def dumpAnnotations(ir: Circuit, optName: Option[File]): File =

	
			Emit the annotations of a circuit
		
		
		:param ir: The circuit containing annotations to be emitted
		
		:param optName: An optional filename (will use s"\${ir.name}.json" otherwise)
		    


	.. chisel:attr:: def dumpProto(c: Circuit, optFile: Option[File]): File =

	
			Dump the elaborated Circuit to ProtoBuf.
		
		If no File is given as input, it will dump to a default filename based on the name of the
		top Module.
		
		
		:param c: Elaborated Chisel Circuit.
		
		:param optFile: Optional File to dump to. If unspecified, defaults to "<topmodule>.pb".
		:return: The File the circuit was dumped to.
		    


	.. chisel:attr:: def execute(optionsManager: ExecutionOptionsManager with HasChiselExecutionOptions with HasFirrtlOptions, dut: () => RawModule): ChiselExecutionResult =

	
			Run the chisel3 compiler and possibly the firrtl compiler with options specified
		
		
		:param optionsManager: The options specified
		
		:param dut:                    The device under test
		:return:                       An execution result with useful stuff, or failure with message
		    


	.. chisel:attr:: def execute(args: Array[String], dut: () => RawModule): ChiselExecutionResult =

	
			Run the chisel3 compiler and possibly the firrtl compiler with options specified via an array of Strings
		
		
		:param args:   The options specified, command line style
		
		:param dut:    The device under test
		:return:       An execution result with useful stuff, or failure with message
		    


	.. chisel:attr:: def main(args: Array[String])

	
			This is just here as command line way to see what the options are
		It will not successfully run
		TODO: Look into dynamic class loading as way to make this main useful
		
		
		:param args: unused args
		    


ChiselExecutionOptions.scala
----------------------------
.. chisel:attr:: case class ChiselExecutionOptions(runFirrtlCompiler: Boolean = true, printFullStackTrace: Boolean = false

		Options that are specific to chisel.
	
	
	:param runFirrtlCompiler: when true just run chisel, when false run chisel then compile its output with firrtl
	
	:note: this extends FirrtlExecutionOptions which extends CommonOptions providing easy access to down chain options
  

.. chisel:attr:: case class ChiselExecutionOptions(runFirrtlCompiler: Boolean


.. chisel:attr:: trait HasChiselExecutionOptions


compatibility.scala
-------------------
.. chisel:attr:: package object Chisel


	.. chisel:attr:: def fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Creates an new instance of this type, unpacking the input Bits into	structured data.
		
		This performs the inverse operation of toBits.
		
		
		:note: does NOT assign to the object this is called on, instead creates
			and returns a NEW object (useful in a clone-and-assign scenario)
		
		:note: does NOT check bit widths, may drop bits during assignment
		
		:note: what fromBits assigs to must have known widths
	      


	.. chisel:attr:: def fill[T <: Data](n: Int)(gen: => T)(implicit compileOptions: CompileOptions): Vec[T]

	
		Creates a new :chisel:reref:`Vec`  of length `n` composed of the result of the given	function repeatedly applied.
		
		
		:param n:   number of elements (and the number of times the function is
			called)
		
		:param gen: function that generates the :chisel:reref:`Data`  that becomes the output
			element
		      


	.. chisel:attr:: def do_apply[T <: Data](elts: Seq[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T]

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_apply[T <: Data](elt0: T, elts: T*) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T]

	
		@group SourceInfoTransformMacro 
	


	.. chisel:attr:: def do_tabulate[T <: Data](n: Int)(gen: (Int) => T) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T]

	
		@group SourceInfoTransformMacro 
	


.. chisel:attr:: trait UIntFactory extends chisel3.UIntFactory

	This contains literal constructor factory methods that are deprecated as of Chisel3.    

	.. chisel:attr:: def apply(n: String): UInt

	
		Create a UInt literal with inferred width. 
	


	.. chisel:attr:: def apply(n: String, width: Int): UInt

	
		Create a UInt literal with fixed width. 
	


	.. chisel:attr:: def apply(value: BigInt, width: Width): UInt

	
		Create a UInt literal with specified width. 
	


	.. chisel:attr:: def apply(value: BigInt, width: Int): UInt

	
		Create a UInt literal with fixed width. 
	


	.. chisel:attr:: def apply(dir: Option[Direction] = None, width: Int): UInt = apply(width.W)

	
		Create a UInt with a specified width - compatibility with Chisel2. 
	


	.. chisel:attr:: def apply(value: BigInt): UInt

	
		Create a UInt literal with inferred width.- compatibility with Chisel2. 
	


	.. chisel:attr:: def apply(dir: Direction, width: Int): UInt

	
		Create a UInt with a specified direction and width - compatibility with Chisel2. 
	


	.. chisel:attr:: def apply(dir: Direction): UInt

	
		Create a UInt with a specified direction, but unspecified width - compatibility with Chisel2. 
	


	.. chisel:attr:: def width(width: Int): UInt

	
		Create a UInt with a specified width 
	


	.. chisel:attr:: def width(width: Width): UInt

	
		Create a UInt port with specified width. 
	


.. chisel:attr:: trait SIntFactory extends chisel3.SIntFactory

	This contains literal constructor factory methods that are deprecated as of Chisel3.    

	.. chisel:attr:: def width(width: Int): SInt

	
		Create a SInt type or port with fixed width. 
	


	.. chisel:attr:: def width(width: Width): SInt

	
		Create an SInt type with specified width. 
	


	.. chisel:attr:: def apply(value: BigInt): SInt

	
		Create an SInt literal with inferred width. 
	


	.. chisel:attr:: def apply(value: BigInt, width: Int): SInt

	
		Create an SInt literal with fixed width. 
	


	.. chisel:attr:: def apply(value: BigInt, width: Width): SInt

	
		Create an SInt literal with specified width. 
	


	.. chisel:attr:: def apply(dir: Option[Direction] = None, width: Int): SInt = apply(width.W)

	
		Create a SInt with a specified width - compatibility with Chisel2. 
	


	.. chisel:attr:: def apply(dir: Direction, width: Int): SInt

	
		Create a SInt with a specified direction and width - compatibility with Chisel2. 
	


	.. chisel:attr:: def apply(dir: Direction): SInt

	
		Create a SInt with a specified direction, but unspecified width - compatibility with Chisel2. 
	


.. chisel:attr:: trait BoolFactory extends chisel3.BoolFactory

	This contains literal constructor factory methods that are deprecated as of Chisel3.    

	.. chisel:attr:: def apply(x: Boolean): Bool

	
		Creates Bool literal.      


	.. chisel:attr:: def apply(dir: Direction): Bool =

	
		Create a UInt with a specified direction and width - compatibility with Chisel2. 
	


	.. chisel:attr:: def apply[T <: Data](t: T = null, next: T = null, init: T = null) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =

	
		Creates a register with optional next and initialization values.	
		
		:param t:: data type for the register
		
		:param next:: new value register is to be updated with every cycle (or
			empty to not update unless assigned to using the := operator)
		
		:param init:: initialization value on reset (or empty for uninitialized,
			where the register value persists across a reset)
			
		
		:note: this may result in a type error if called from a type parameterized
			function, since the Scala compiler isn't smart enough to know that null
			is a valid value. In those cases, you can either use the outType only Reg
			constructor or pass in `null.asInstanceOf[T]`.
	      


.. chisel:attr:: object log2Up

	Compute the log2 rounded up with min value of 1 


.. chisel:attr:: object log2Down

	Compute the log2 rounded down with min value of 1 


	.. chisel:attr:: def apply[T <: Bits](nodeType: T, n: Int): List[T] =

	
		Returns n unique values of the specified type. Can be used with unpacking to define enums.	
		nodeType must be of UInt type (note that Bits() creates a UInt) with unspecified width.
		
		
		.. code-block:: scala 
	
			 val state_on :: state_off :: Nil = Enum(UInt(), 2)
			 val current_state = UInt()
			 switch (current_state) {
			   is (state_on) {
			      ...
			   }
			   if (state_off) {
			      ...
			   }
			 }
		
	      


	.. chisel:attr:: def apply[T <: Bits](nodeType: T, l: Symbol *): Map[Symbol, T] =

	
		An old Enum API that returns a map of symbols to UInts.	
		Unlike the new list-based Enum, which can be unpacked into vals that the compiler
		understands and can check, map accesses can't be compile-time checked and typos may not be
		caught until runtime.
		
		Despite being deprecated, this is not to be removed from the compatibility layer API.
		Deprecation is only to nag users to do something safer.
	      


	.. chisel:attr:: def apply[T <: Bits](nodeType: T, l: List[Symbol]): Map[Symbol, T] =

	
		An old Enum API that returns a map of symbols to UInts.	
		Unlike the new list-based Enum, which can be unpacked into vals that the compiler
		understands and can check, map accesses can't be compile-time checked and typos may not be
		caught until runtime.
		
		Despite being deprecated, this is not to be removed from the compatibility layer API.
		Deprecation is only to nag users to do something safer.
	      


.. chisel:attr:: object experimental

	Package for experimental features, which may have their API changed, be removed, etc.	
	Because its contents won't necessarily have the same level of stability and support as
	non-experimental, you must explicitly import this package to use its contents.
    

