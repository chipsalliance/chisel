----------------------------------------
src/main/scala/chisel3/util/experimental
----------------------------------------

.. toctree::


LoadMemoryTransform.scala
-------------------------
.. chisel:attr:: case class ChiselLoadMemoryAnnotation[T <: Data](target:      MemBase[T], fileName:    String, hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex ) extends ChiselAnnotation with RunFirrtlTransform

	This is the annotation created when using :chisel:reref:`loadMemoryFromFile` , it records the memory, the load file	and the format of the file.
	
	:param target:        memory to load
	
	:param fileName:      name of input file
	
	:param hexOrBinary:   use \$readmemh or \$readmemb, i.e. hex or binary text input, default is hex
	  

.. chisel:attr:: case class ChiselLoadMemoryAnnotation[T <: Data](target:      MemBase[T], fileName:    String, hexOrBinary: MemoryLoadFileType.FileType


.. chisel:attr:: object loadMemoryFromFile

	:chisel:reref:`loadMemoryFromFile`  is an annotation generator that helps with loading a memory from a text file. This relies on	Verilator and Verilog's `\$readmemh` or `\$readmemb`. The :chisel:reref:`https://github.com/freechipsproject/treadle Treadle
	backend`  can also recognize this annotation and load memory at run-time.
	
	This annotation, when the FIRRTL compiler runs, triggers the :chisel:reref:`LoadMemoryTransform` . That will add Verilog
	directives to enable the specified memories to be initialized from files.
	
	==Example module==
	
	Consider a simple Module containing a memory:
	
	.. code-block:: scala 

		 import chisel3._
		 class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
		   val io = IO(new Bundle {
		     val address = Input(UInt(memoryType.getWidth.W))
		     val value   = Output(memoryType)
		   })
		   val memory = Mem(memoryDepth, memoryType)
		   io.value := memory(io.address)
		 }
	
	
	==Above module with annotation==
	
	To load this memory from the file `/workspace/workdir/mem1.hex.txt` just add an import and annotate the memory:
	
	.. code-block:: scala 

		 import chisel3._
		 import chisel3.util.experimental.loadMemoryFromFile   // <<-- new import here
		 class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
		   val io = IO(new Bundle {
		     val address = Input(UInt(memoryType.getWidth.W))
		     val value   = Output(memoryType)
		   })
		   val memory = Mem(memoryDepth, memoryType)
		   io.value := memory(io.address)
		   loadMemoryFromFile(memory, "/workspace/workdir/mem1.hex.txt")  // <<-- Note the annotation here
		 }
	
	
	==Example file format==
	
	A memory file should consist of ASCII text in either hex or binary format. The following example shows such a
	file formatted to use hex:
	
	.. code-block:: scala 

		   0
		   7
		   d
		  15
	
	
	A binary file can be similarly constructed.
	
	@see
	:chisel:reref:`https://github.com/freechipsproject/chisel3/tree/master/src/test/scala/chiselTests/LoadMemoryFromFileSpec.scala
	LoadMemoryFromFileSpec.scala`  in the test suite for additional examples.
	@see Chisel3 Wiki entry on
	:chisel:reref:`https://github.com/freechipsproject/chisel3/wiki/Chisel-Memories#loading-memories-in-simulation "Loading Memories
	in Simulation"`
  

	.. chisel:attr:: def apply[T <: Data](memory: MemBase[T], fileName: String, hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex ): Unit =

	
		Annotate a memory such that it can be initialized using a file	
		:param memory: the memory
		
		:param filename: the file used for initialization
		
		:param hexOrBinary: whether the file uses a hex or binary number representation
		    


.. chisel:attr:: class LoadMemoryTransform extends Transform

	This transform only is activated if Verilog is being generated (determined by presence of the proper emit	annotation) when activated it creates additional Verilog files that contain modules bound to the modules that
	contain an initializable memory
	
	Currently the only non-Verilog based simulation that can support loading memory from a file is treadle but it does
	not need this transform to do that.
  

	.. chisel:attr:: def run(circuit: Circuit, annotations: AnnotationSeq): Circuit =

	
		run the pass	
		:param circuit: the circuit
		
		:param annotations: all the annotations
			@return
		    


BoringUtils.scala
-----------------
.. chisel:attr:: class BoringUtilsException(message: String) extends Exception(message)

	An exception related to BoringUtils	
	:param message: the exception message
	  

.. chisel:attr:: object BoringUtils

	Utilities for generating synthesizable cross module references that "bore" through the hierarchy. The underlying	cross module connects are handled by FIRRTL's Wiring Transform.
	
	Consider the following exmple where you want to connect a component in one module to a component in another. Module
	`Constant` has a wire tied to `42` and `Expect` will assert unless connected to `42`:
	
	.. code-block:: scala 

		 class Constant extends Module {
		   val io = IO(new Bundle{})
		   val x = Wire(UInt(6.W))
		   x := 42.U
		 }
		 class Expect extends Module {
		   val io = IO(new Bundle{})
		   val y = Wire(UInt(6.W))
		   y := 0.U
		   // This assertion will fail unless we bore!
		   chisel3.assert(y === 42.U, "y should be 42 in module Expect")
		 }
	
	
	We can then connect `x` to `y` using :chisel:reref:`BoringUtils`  without modifiying the Chisel IO of `Constant`, `Expect`, or
	modules that may instantiate them. There are two approaches to do this:
	
	1. Hierarchical boring using :chisel:reref:`BoringUtils.bore`
	
	2. Non-hierarchical boring using :chisel:reref:`BoringUtils.addSink` /:chisel:reref:`BoringUtils.addSource`
	
	===Hierarchical Boring===
	
	Hierarchcical boring involves connecting one sink instance to another source instance in a parent module. Below,
	module `Top` contains an instance of `Cosntant` and `Expect`. Using :chisel:reref:`BoringUtils.bore` , we can connect
	`constant.x` to `expect.y`.
	
	
	.. code-block:: scala 

		 class Top extends Module {
		   val io = IO(new Bundle{})
		   val constant = Module(new Constant)
		   val expect = Module(new Expect)
		   BoringUtils.bore(constant.x, Seq(expect.y))
		 }
	
	
	===Non-hierarchical Boring===
	
	Non-hierarchical boring involves connections from sources to sinks that cannot see each other. Here, `x` is
	described as a source and given a name, `uniqueId`, and `y` is described as a sink with the same name. This is
	equivalent to the hierarchical boring example above, but requires no modifications to `Top`.
	
	
	.. code-block:: scala 

		 class Constant extends Module {
		   val io = IO(new Bundle{})
		   val x = Wire(UInt(6.W))
		   x := 42.U
		   BoringUtils.addSource(x, "uniqueId")
		 }
		 class Expect extends Module {
		   val io = IO(new Bundle{})
		   val y = Wire(UInt(6.W))
		   y := 0.U
		   // This assertion will fail unless we bore!
		   chisel3.assert(y === 42.U, "y should be 42 in module Expect")
		   BoringUtils.addSink(y, "uniqueId")
		 }
		 class Top extends Module {
		   val io = IO(new Bundle{})
		   val constant = Module(new Constant)
		   val expect = Module(new Expect)
		 }
	
	
	==Comments==
	
	Both hierarchical and non-hierarchical boring emit FIRRTL annotations that describe sources and sinks. These are
	matched by a `name` key that indicates they should be wired together. Hierarhical boring safely generates this name
	automatically. Non-hierarchical boring unsafely relies on user input to generate this name. Use of non-hierarchical
	naming may result in naming conflicts that the user must handle.
	
	The automatic generation of hierarchical names relies on a global, mutable namespace. This is currently persistent
	across circuit elaborations.
  

	.. chisel:attr:: def addSource(component: NamedComponent, name: String, disableDedup: Boolean = false, uniqueName: Boolean = false): String =

	
		Add a named source cross module reference	
		:param component: source circuit component
		
		:param name: unique identifier for this source
		
		:param disableDedup: disable dedupblication of this source component (this should be true if you are trying to wire
			from specific identical sources differently)
		
		:param uniqueName: if true, this will use a non-conflicting name from the global namespace
		:return: the name used
		
		:note: if a uniqueName is not specified, the returned name may differ from the user-provided name
	    


	.. chisel:attr:: def addSink(component: InstanceId, name: String, disableDedup: Boolean = false, forceExists: Boolean = false): Unit =

	
		Add a named sink cross module reference. Multiple sinks may map to the same source.	
		:param component: sink circuit component
		
		:param name: unique identifier for this sink that must resolve to
		
		:param disableDedup: disable deduplication of this sink component (this should be true if you are trying to wire
			specific, identical sinks differently)
		
		:param forceExists: if true, require that the provided `name` paramater already exists in the global namespace
			@throws BoringUtilsException if name is expected to exist and itdoesn't
		    


	.. chisel:attr:: def bore(source: Data, sinks: Seq[Data]): String =

	
		Connect a source to one or more sinks	
		:param source: a source component
		
		:param sinks: one or more sink components
		:return: the name of the signal used to connect the source to the
			sinks
		
		:note: the returned name will be based on the name of the source
			component
	    


Inline.scala
------------
.. chisel:attr:: * class Foo extends Module with Internals with InlineInstance with HasSub *

	Inlines an instance of a module	
	
	.. code-block:: scala 

		 trait Internals { this: Module =>
		   val io = IO(new Bundle{ val a = Input(Bool()) })
		 }
		 class Sub extends Module with Internals
		 trait HasSub { this: Module with Internals =>
		   val sub = Module(new Sub)
		   sub.io.a := io.a
		 }
		 /* InlineInstance is mixed directly into Foo's definition. Every instance
  *  * of this will be inlined. 

.. chisel:attr:: trait InlineInstance


.. chisel:attr:: * class Top extends Module with Internals

	Flattens an instance of a module	
	
	.. code-block:: scala 

		 trait Internals { this: Module =>
		   val io = IO(new Bundle{ val a = Input(Bool()) })
		 }
		 class Foo extends Module with Internals with FlattenInstance
		 class Bar extends Module with Internals {
		   val baz = Module(new Baz)
		   baz.io.a := io.a
		 }
		 class Baz extends Module with Internals
		 /* The resulting instances will be:
		      - Top
		      - Top.x
		      - Top.y
		      - Top.z
  *      - Top.z.baz 

.. chisel:attr:: trait FlattenInstance


