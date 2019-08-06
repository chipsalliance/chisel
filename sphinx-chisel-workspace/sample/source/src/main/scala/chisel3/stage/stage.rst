----------------------------
src/main/scala/chisel3/stage
----------------------------

.. toctree::
	phases/phases.rst


ChiselStage.scala
-----------------
.. chisel:attr:: class ChiselStage extends Stage


ChiselOptions.scala
-------------------
.. chisel:attr:: class ChiselOptions private[stage] (val runFirrtlCompiler:   Boolean


ChiselCli.scala
---------------
.. chisel:attr:: trait ChiselCli


ChiselAnnotations.scala
-----------------------
.. chisel:attr:: sealed trait ChiselOption extends Unserializable

	Mixin that indicates that this is an :chisel:reref:`firrtl.annotations.Annotation`  used to generate a :chisel:reref:`ChiselOptions`  view.  

.. chisel:attr:: case object NoRunFirrtlCompilerAnnotation extends NoTargetAnnotation with ChiselOption with HasShellOptions

	Disable the execution of the FIRRTL compiler by Chisel  

.. chisel:attr:: case object PrintFullStackTraceAnnotation extends NoTargetAnnotation with ChiselOption with HasShellOptions

	On an exception, this will cause the full stack trace to be printed as opposed to a pruned stack trace.  

.. chisel:attr:: case class ChiselGeneratorAnnotation(gen: () => RawModule) extends NoTargetAnnotation with Unserializable

	An :chisel:reref:`firrtl.annotations.Annotation`  storing a function that returns a Chisel module	
	:param gen: a generator function
	  

	.. chisel:attr:: def elaborate: ChiselCircuitAnnotation

	
		Run elaboration on the Chisel module generator function stored by this :chisel:reref:`firrtl.annotations.Annotation`     


.. chisel:attr:: object ChiselGeneratorAnnotation extends HasShellOptions


	.. chisel:attr:: def apply(name: String): ChiselGeneratorAnnotation =

	
		Construct a :chisel:reref:`ChiselGeneratorAnnotation`  with a generator function that will try to construct a Chisel Module	from using that Module's name. The Module must both exist in the class path and not take parameters.
		
		:param name: a module name
			@throws firrtl.options.OptionsException if the module name is not found or if no parameterless constructor for
			that Module is found
		    


.. chisel:attr:: case class ChiselCircuitAnnotation(circuit: Circuit) extends NoTargetAnnotation with ChiselOption

	Stores a Chisel Circuit	
	:param circuit: a Chisel Circuit
	  

.. chisel:attr:: case class ChiselCircuitAnnotation(circuit: Circuit) extends NoTargetAnnotation with ChiselOption  case class ChiselOutputFileAnnotation(file: String) extends NoTargetAnnotation with ChiselOption


.. chisel:attr:: case class ChiselOutputFileAnnotation(file: String) extends NoTargetAnnotation with ChiselOption  object ChiselOutputFileAnnotation extends HasShellOptions


.. chisel:attr:: object ChiselOutputFileAnnotation extends HasShellOptions


package.scala
-------------
.. chisel:attr:: package object stage


