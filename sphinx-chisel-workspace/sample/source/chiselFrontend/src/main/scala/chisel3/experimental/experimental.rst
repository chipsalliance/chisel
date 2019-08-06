--------------------------------------------------
chiselFrontend/src/main/scala/chisel3/experimental
--------------------------------------------------

.. toctree::


package.scala
-------------
.. chisel:attr:: package object experimental

	Package for experimental features, which may have their API changed, be removed, etc.	
	Because its contents won't necessarily have the same level of stability and support as
	non-experimental, you must explicitly import this package to use its contents.
  

	.. chisel:attr:: def apply(proto: BaseModule)(implicit sourceInfo: chisel3.internal.sourceinfo.SourceInfo, compileOptions: CompileOptions): ClonePorts =

	
		Clones an existing module and returns a record of all its top-level ports.	Each element of the record is named with a string matching the
		corresponding port's name and shares the port's type.
		
		.. code-block:: scala 
	
			 val q1 = Module(new Queue(UInt(32.W), 2))
			 val q2_io = CloneModuleAsRecord(q1)("io").asInstanceOf[q1.io.type]
			 q2_io.enq <> q1.io.deq
		
	      


	.. chisel:attr:: def range(args: Any*): (NumericBound[Int], NumericBound[Int])

	
		Specifies a range using mathematical range notation. Variables can be interpolated using	standard string interpolation syntax.
		
		.. code-block:: scala 
	
			 UInt(range"[0, 2)")
			 UInt(range"[0, \$myInt)")
			 UInt(range"[0, \${myInt + 2})")
		
	      


