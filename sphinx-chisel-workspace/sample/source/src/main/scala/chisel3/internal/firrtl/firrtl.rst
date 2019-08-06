--------------------------------------
src/main/scala/chisel3/internal/firrtl
--------------------------------------

.. toctree::


Emitter.scala
-------------
.. chisel:attr:: private[chisel3] object Emitter


.. chisel:attr:: private class Emitter(circuit: Circuit)


	.. chisel:attr:: private def moduleDecl(m: Component): String

	
		Generates the FIRRTL module declaration.    


	.. chisel:attr:: private def moduleDefn(m: Component): String =

	
		Generates the FIRRTL module definition.    


	.. chisel:attr:: private def emit(m: Component): String =

	
		Returns the FIRRTL declaration and body of a module, or nothing if it's a	duplicate of something already emitted (on the basis of simple string
		matching).
	    


	.. chisel:attr:: private def processWhens(cmds: Seq[Command]): Seq[Command] =

	
		Preprocess the command queue, marking when/elsewhen statements	that have no alternatives (elsewhens or otherwise). These
		alternative-free statements reset the indent level to the
		enclosing block upon emission.
	    


Converter.scala
---------------
.. chisel:attr:: private[chisel3] object Converter


	.. chisel:attr:: def convertSimpleCommand(cmd: Command, ctx: Component): Option[fir.Statement]

	
		Convert Commands that map 1:1 to Statements 
	


.. chisel:attr:: private case class WhenFrame(when: fir.Conditionally, outer: Queue[fir.Statement], alt: Boolean)

	Internal datastructure to help translate Chisel's flat Command structure to FIRRTL's AST	
	In particular, when scoping is translated from flat with begin end to a nested datastructure
	
	
	:param when: Current when Statement, holds info, condition, and consequence as they are
		available
	
	:param outer: Already converted Statements that precede the current when block in the scope in
		which the when is defined (ie. 1 level up from the scope inside the when)
	
	:param alt: Indicates if currently processing commands in the alternate (else) of the when scope
	    

	.. chisel:attr:: def convert(cmds: Seq[Command], ctx: Component): fir.Statement =

	
		Convert Chisel IR Commands into FIRRTL Statements	
		
		:note: ctx is needed because references to ports translate differently when referenced within
			the module in which they are defined vs. parent modules
		
		:param cmds: Chisel IR Commands to convert
		
		:param ctx: Component (Module) context within which we are translating
		:return: FIRRTL Statement that is equivalent to the input cmds
		    


