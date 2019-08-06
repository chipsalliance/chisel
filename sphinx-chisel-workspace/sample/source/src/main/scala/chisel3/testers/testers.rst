------------------------------
src/main/scala/chisel3/testers
------------------------------

.. toctree::


package.scala
-------------
.. chisel:attr:: package object testers

	The testers package provides the basic interface for chisel testers.	
  

BasicTester.scala
-----------------
.. chisel:attr:: class BasicTester extends Module()


	.. chisel:attr:: def stop()(implicit sourceInfo: SourceInfo)

	
		Ends the test reporting success.	
		Does not fire when in reset (defined as the encapsulating Module's
		reset). If your definition of reset is not the encapsulating Module's
		reset, you will need to gate this externally.
	    


	.. chisel:attr:: def finish(): Unit =

	
		The finish method provides a hook that subclasses of BasicTester can use to	alter a circuit after their constructor has been called.
		For example, a specialized tester subclassing BasicTester could override finish in order to
		add flow control logic for a decoupled io port of a device under test
	    


TesterDriver.scala
------------------
.. chisel:attr:: object TesterDriver extends BackendCompilationUtilities


	.. chisel:attr:: def execute(t: () => BasicTester, additionalVResources: Seq[String] = Seq()): Boolean =

	
		For use with modules that should successfully be elaborated by the    * frontend, and which can be turned into executables with assertions. 


	.. chisel:attr:: def finishWrapper(test: () => BasicTester): () => BasicTester =

	
			Calls the finish method of an BasicTester or a class that extends it.
		The finish method is a hook for code that augments the circuit built in the constructor.
	    


