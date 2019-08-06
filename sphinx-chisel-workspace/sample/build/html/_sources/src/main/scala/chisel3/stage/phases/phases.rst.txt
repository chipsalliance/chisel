-----------------------------------
src/main/scala/chisel3/stage/phases
-----------------------------------

.. toctree::


Checks.scala
------------
.. chisel:attr:: class Checks extends Phase

	Sanity checks an :chisel:reref:`firrtl.AnnotationSeq`  before running the main :chisel:reref:`firrtl.options.Phase` s of	:chisel:reref:`chisel3.stage.ChiselStage` .
  

Emitter.scala
-------------
.. chisel:attr:: class Emitter extends Phase

	Emit a :chisel:reref:`chisel3.stage.ChiselCircuitAnnotation`  to a file if a :chisel:reref:`chisel3.stage.ChiselOutputFileAnnotation`  is	present. A deleted :chisel:reref:`firrtl.EmittedFirrtlCircuitAnnotation`  is added.
	
	@todo This should be switched to support correct emission of multiple circuits to multiple files. The API should
	likely mirror how the :chisel:reref:`firrtl.stage.phases.Compiler`  parses annotations into "global" annotations and
	left-associative per-circuit annotations.
	@todo The use of the deleted :chisel:reref:`firrtl.EmittedFirrtlCircuitAnnotation`  is a kludge to provide some breadcrumbs such
	that the emitted CHIRRTL can be provided back to the old Driver. This should be removed or a better solution
	developed.
  

AddImplicitOutputAnnotationFile.scala
-------------------------------------
.. chisel:attr:: class AddImplicitOutputAnnotationFile extends Phase

	Adds an :chisel:reref:`firrtl.options.OutputAnnotationFileAnnotation`  if one does not exist. This replicates old behavior where	an output annotation file was always written.
  

AddImplicitOutputFile.scala
---------------------------
.. chisel:attr:: class AddImplicitOutputFile extends Phase

	Add a output file for a Chisel circuit, derived from the top module in the circuit, if no	:chisel:reref:`ChiselOutputFileAnnotation`  already exists.
  

DriverCompatibility.scala
-------------------------
.. chisel:attr:: object DriverCompatibility

	This provides components of a compatibility wrapper around Chisel's deprecated :chisel:reref:`chisel3.Driver` .	
	Primarily, this object includes :chisel:reref:`firrtl.options.Phase Phase` s that generate :chisel:reref:`firrtl.annotations.Annotation` s
	derived from the deprecated :chisel:reref:`firrtl.stage.phases.DriverCompatibility.TopNameAnnotation` .
  

.. chisel:attr:: private[chisel3] class AddImplicitOutputFile extends Phase

	Adds a :chisel:reref:`ChiselOutputFileAnnotation`  derived from a :chisel:reref:`TopNameAnnotation`  if no :chisel:reref:`ChiselOutputFileAnnotation` 	already exists. If no :chisel:reref:`TopNameAnnotation`  exists, then no :chisel:reref:`firrtl.stage.OutputFileAnnotation`  is added. ''This is not a
	replacement for :chisel:reref:`chisel3.stage.phases.AddImplicitOutputFile AddImplicitOutputFile`  as this only adds an output
	file based on a discovered top name and not on a discovered elaborated circuit.'' Consequently, this will provide
	the correct behavior before a circuit has been elaborated.
	
	:note: the output suffix is unspecified and will be set by the underlying :chisel:reref:`firrtl.EmittedComponent`
    

.. chisel:attr:: private[chisel3] class AddImplicitOutputAnnotationFile extends Phase

	If a :chisel:reref:`firrtl.options.OutputAnnotationFileAnnotation`  does not exist, this adds one derived from a	:chisel:reref:`TopNameAnnotation` . ''This is not a replacement for :chisel:reref:`chisel3.stage.phases.AddImplicitOutputAnnotationFile`  as
	this only adds an output annotation file based on a discovered top name.'' Consequently, this will provide the
	correct behavior before a circuit has been elaborated.
	
	:note: the output suffix is unspecified and will be set by :chisel:reref:`firrtl.options.phases.WriteOutputAnnotations`
    

.. chisel:attr:: private[chisel3] class DisableFirrtlStage extends Phase

	Disables the execution of :chisel:reref:`firrtl.stage.FirrtlStage` . This can be used to call :chisel:reref:`chisel3.stage.ChiselStage`  and	guarantee that the FIRRTL compiler will not run. This is necessary for certain :chisel:reref:`chisel3.Driver`  compatibility
	situations where you need to do something between Chisel compilation and FIRRTL compilations, e.g., update a
	mutable data structure.
    

.. chisel:attr:: private[chisel3] class MutateOptionsManager(optionsManager: ExecutionOptionsManager with HasChiselExecutionOptions with HasFirrtlOptions) extends Phase

	Mutate an input :chisel:reref:`firrtl.ExecutionOptionsManager`  based on information encoded in an :chisel:reref:`firrtl.AnnotationSeq` .	This is intended to be run between :chisel:reref:`chisel3.stage.ChiselStage ChiselStage`  and :chisel:reref:`firrtl.stage.FirrtlStage`  if
	you want to have backwards compatibility with an :chisel:reref:`firrtl.ExecutionOptionsManager` .
    

MaybeFirrtlStage.scala
----------------------
.. chisel:attr:: class MaybeFirrtlStage extends Phase

	Run :chisel:reref:`firrtl.stage.FirrtlStage`  if a :chisel:reref:`chisel3.stage.NoRunFirrtlCompilerAnnotation`  is not present.  

Elaborate.scala
---------------
.. chisel:attr:: class Elaborate extends Phase

	Elaborate all :chisel:reref:`chisel3.stage.ChiselGeneratorAnnotation` s into :chisel:reref:`chisel3.stage.ChiselCircuitAnnotation` s.  

	.. chisel:attr:: def transform(annotations: AnnotationSeq): AnnotationSeq

	
			@todo Change this to print to STDERR (`Console.err.println`)
	    


Convert.scala
-------------
.. chisel:attr:: class Convert extends Phase

	This prepares a :chisel:reref:`ChiselCircuitAnnotation`  for compilation with FIRRTL. This does three things:	- Uses :chisel:reref:`chisel3.internal.firrtl.Converter`  to generate a :chisel:reref:`FirrtlCircuitAnnotation`
	- Extracts all :chisel:reref:`firrtl.annotations.Annotation` s from the :chisel:reref:`chisel3.internal.firrtl.Circuit`
	- Generates any needed :chisel:reref:`RunFirrtlTransformAnnotation` s from extracted :chisel:reref:`firrtl.annotations.Annotation` s
  

