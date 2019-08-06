----------------------------------------------
chiselFrontend/src/main/scala/chisel3/internal
----------------------------------------------

.. toctree::
	firrtl/firrtl.rst


MonoConnect.scala
-----------------
.. chisel:attr:: private[chisel3] object MonoConnect

		MonoConnect.connect executes a mono-directional connection element-wise.
	
	Note that this isn't commutative. There is an explicit source and sink
	already determined before this function is called.
	
	The connect operation will recurse down the left Data (with the right Data).
	An exception will be thrown if a movement through the left cannot be matched
	in the right. The right side is allowed to have extra Record fields.
	Vecs must still be exactly the same size.
	
	See elemConnect for details on how the root connections are issued.
	
	Note that a valid sink must be writable so, one of these must hold:
	- Is an internal writable node (Reg or Wire)
	- Is an output of the current module
	- Is an input of a submodule of the current module
	
	Note that a valid source must be readable so, one of these must hold:
	- Is an internal readable node (Reg, Wire, Op)
	- Is a literal
	- Is a port of the current module or submodule of the current module


	.. chisel:attr:: def connect(sourceInfo: SourceInfo, connectCompileOptions: CompileOptions, sink: Data, source: Data, context_mod: RawModule): Unit

	
		This function is what recursively tries to connect a sink and source together	
		There is some cleverness in the use of internal try-catch to catch exceptions
		during the recursive decent and then rethrow them with extra information added.
		This gives the user a 'path' to where in the connections things went wrong.
	  


Builder.scala
-------------
.. chisel:attr:: private[chisel3] class Namespace(keywords: Set[String])


.. chisel:attr:: private[chisel3] object Namespace


	.. chisel:attr:: def empty: Namespace

	
		Constructs an empty Namespace 
	


.. chisel:attr:: private[chisel3] class IdGen


.. chisel:attr:: trait InstanceId

	Public API to access Node/Signal names.	currently, the node's name, the full path name, and references to its parent Module and component.
	These are only valid once the design has been elaborated, and should not be used during its construction.
  

	.. chisel:attr:: def toNamed: Named

	
		Returns a FIRRTL Named that refers to this object in the elaborated hardware graph 
	


.. chisel:attr:: private[chisel3] trait HasId extends InstanceId


.. chisel:attr:: private[chisel3] trait NamedComponent extends HasId

	Holds the implementation of toNamed for Data and MemBase 


	.. chisel:attr:: final def toNamed: ComponentName

	
		Returns a FIRRTL ComponentName that references this object	
		:note: Should not be called until circuit elaboration is complete
	    


.. chisel:attr:: private[chisel3] class ChiselContext()


.. chisel:attr:: private[chisel3] class DynamicContext()


.. chisel:attr:: private[chisel3] object Builder


	.. chisel:attr:: def nameRecursively(prefix: String, nameMe: Any, namer: (HasId, String) => Unit): Unit

	
		Recursively suggests names to supported "container" classes	Arbitrary nestings of supported classes are allowed so long as the
		innermost element is of type HasId
		(Note: Map is Iterable[Tuple2[_,_`  and thus excluded)
	    


	.. chisel:attr:: def exception(m: => String): Unit =

	
		Record an exception as an error, and throw it.	
		
		:param m: exception message
		    


.. chisel:attr:: object DynamicNamingStack

	Allows public access to the naming stack in Builder / DynamicContext, and handles invocations	outside a Builder context.
	Necessary because naming macros expand in user code and don't have access into private[chisel3]
	objects.
  

.. chisel:attr:: private[chisel3] object castToInt

	Casts BigInt to Int, issuing an error when the input isn't representable. 


Binding.scala
-------------
.. chisel:attr:: object requireIsHardware

	Requires that a node is hardware ("bound")  

.. chisel:attr:: object requireIsChiselType

	Requires that a node is a chisel type (not hardware, "unbound")  

.. chisel:attr:: private[chisel3] sealed abstract class BindingDirection private[chisel3] object BindingDirection


.. chisel:attr:: private[chisel3] object BindingDirection


.. chisel:attr:: case object Internal extends BindingDirection

	Internal type or wire    

.. chisel:attr:: case object Output extends BindingDirection

	Module port with output direction    

.. chisel:attr:: case object Input extends BindingDirection

	Module port with input direction    

	.. chisel:attr:: def from(binding: TopBinding, direction: ActualDirection): BindingDirection =

	
		Determine the BindingDirection of an Element given its top binding and resolved direction.    


.. chisel:attr:: sealed trait Binding


.. chisel:attr:: sealed trait TopBinding extends Binding


.. chisel:attr:: sealed trait UnconstrainedBinding extends TopBinding


.. chisel:attr:: sealed trait ConstrainedBinding extends TopBinding


.. chisel:attr:: sealed trait ReadOnlyBinding extends TopBinding


.. chisel:attr:: case class OpBinding(enclosure: RawModule) extends ConstrainedBinding with ReadOnlyBinding case class MemoryPortBinding(enclosure: RawModule) extends ConstrainedBinding case class PortBinding(enclosure: BaseModule) extends ConstrainedBinding


.. chisel:attr:: case class MemoryPortBinding(enclosure: RawModule) extends ConstrainedBinding case class PortBinding(enclosure: BaseModule) extends ConstrainedBinding case class RegBinding(enclosure: RawModule) extends ConstrainedBinding


.. chisel:attr:: case class PortBinding(enclosure: BaseModule) extends ConstrainedBinding case class RegBinding(enclosure: RawModule) extends ConstrainedBinding case class WireBinding(enclosure: RawModule) extends ConstrainedBinding


.. chisel:attr:: case class RegBinding(enclosure: RawModule) extends ConstrainedBinding case class WireBinding(enclosure: RawModule) extends ConstrainedBinding  case class ChildBinding(parent: Data) extends Binding


.. chisel:attr:: case class WireBinding(enclosure: RawModule) extends ConstrainedBinding  case class ChildBinding(parent: Data) extends Binding


.. chisel:attr:: case class ChildBinding(parent: Data) extends Binding


.. chisel:attr:: case class SampleElementBinding[T <: Data](parent: Vec[T]) extends Binding

	Special binding for Vec.sample_element 


.. chisel:attr:: case class DontCareBinding() extends UnconstrainedBinding  sealed trait LitBinding extends UnconstrainedBinding with ReadOnlyBinding


.. chisel:attr:: sealed trait LitBinding extends UnconstrainedBinding with ReadOnlyBinding


.. chisel:attr:: case class ElementLitBinding(litArg: LitArg) extends LitBinding


.. chisel:attr:: case class BundleLitBinding(litMap: Map[Data, LitArg]) extends LitBinding  


Namer.scala
-----------
.. chisel:attr:: sealed trait NamingContextInterface

	Base class for naming contexts, providing the basic API consisting of naming calls and	ability to take descendant naming contexts.
  

	.. chisel:attr:: def name[T](obj: T, name: String): T

	
		Suggest a name (that will be propagated to FIRRTL) for an object, then returns the object	itself (so this can be inserted transparently anywhere).
		Is a no-op (so safe) when applied on objects that aren't named, including non-Chisel data
		types.
	    


	.. chisel:attr:: def namePrefix(prefix: String)

	
		Gives this context a naming prefix (which may be empty, "", for a top-level Module context)	so that actual naming calls (HasId.suggestName) can happen.
		Recursively names descendants, for those whose return value have an associated name.
	    


.. chisel:attr:: object DummyNamer extends NamingContextInterface

	Dummy implementation to allow for naming annotations in a non-Builder context.  

.. chisel:attr:: class NamingContext extends NamingContextInterface

	Actual namer functionality.  

	.. chisel:attr:: def addDescendant(ref: Any, descendant: NamingContext)

	
		Adds a NamingContext object as a descendant - where its contained objects will have names	prefixed with the name given to the reference object, if the reference object is named in the
		scope of this context.
	    


.. chisel:attr:: class NamingStack

	Class for the (global) naming stack object, which provides a way to push and pop naming	contexts as functions are called / finished.
  

	.. chisel:attr:: def pushContext(): NamingContext =

	
		Creates a new naming context, where all items in the context will have their names prefixed	with some yet-to-be-determined prefix from object names in an enclosing scope.
	    


	.. chisel:attr:: def popContext[T <: Any](prefixRef: T, until: NamingContext): Unit =

	
		Called at the end of a function, popping the current naming context, adding it to the	enclosing context's descendants, and passing through the prefix naming reference.
		Every instance of push_context() must have a matching pop_context().
		
		Will assert out if the context being popped isn't the topmost on the stack.
	    


SourceInfo.scala
----------------
.. chisel:attr:: sealed trait SourceInfo

	Abstract base class for generalized source information.  

	.. chisel:attr:: def makeMessage(f: String => String): String

	
		A prettier toString	
		Make a useful message if SourceInfo is available, nothing otherwise
	    


.. chisel:attr:: sealed trait NoSourceInfo extends SourceInfo


.. chisel:attr:: case object UnlocatableSourceInfo extends NoSourceInfo

	For when source info can't be generated because of a technical limitation, like for Reg because	Scala macros don't support named or default arguments.
  

.. chisel:attr:: case object DeprecatedSourceInfo extends NoSourceInfo

	For when source info isn't generated because the function is deprecated and we're lazy.  

.. chisel:attr:: case class SourceLine(filename: String, line: Int, col: Int) extends SourceInfo

	For FIRRTL lines from a Scala source line.  

.. chisel:attr:: object SourceInfoMacro

	Provides a macro that returns the source information at the invocation point.  

.. chisel:attr:: object SourceInfo


Error.scala
-----------
.. chisel:attr:: class ChiselException(message: String, cause: Throwable


	.. chisel:attr:: def trimmedStackTrace: Array[StackTraceElement] =

	
		trims the top of the stack of elements belonging to :chisel:reref:`blacklistPackages` 	then trims the bottom elements until it reaches :chisel:reref:`builderName`
		then continues trimming elements belonging to :chisel:reref:`blacklistPackages`
	    


.. chisel:attr:: private[chisel3] object throwException


.. chisel:attr:: private[chisel3] object ErrorLog

	Records and reports runtime errors and warnings. 


.. chisel:attr:: private[chisel3] class ErrorLog


	.. chisel:attr:: def error(m: => String): Unit

	
		Log an error message 
	


	.. chisel:attr:: def warning(m: => String): Unit

	
		Log a warning message 
	


	.. chisel:attr:: def info(m: String): Unit

	
		Emit an informational message 
	


	.. chisel:attr:: def deprecated(m: => String, location: Option[String]): Unit =

	
		Log a deprecation warning message 
	


	.. chisel:attr:: def checkpoint(): Unit =

	
		Throw an exception if any errors have yet occurred. 
	


	.. chisel:attr:: private def getUserLineNumber =

	
		Returns the best guess at the first stack frame that belongs to user code.    


.. chisel:attr:: private abstract class LogEntry(msg: => String, line: Option[StackTraceElement])


.. chisel:attr:: private class Error(msg: => String, line: Option[StackTraceElement]) extends LogEntry(msg, line)


.. chisel:attr:: private class Warning(msg: => String, line: Option[StackTraceElement]) extends LogEntry(msg, line)


.. chisel:attr:: private class Info(msg: => String, line: Option[StackTraceElement]) extends LogEntry(msg, line)


BiConnect.scala
---------------
.. chisel:attr:: private[chisel3] object BiConnect

		BiConnect.connect executes a bidirectional connection element-wise.
	
	Note that the arguments are left and right (not source and sink) so the
	intent is for the operation to be commutative.
	
	The connect operation will recurse down the left Data (with the right Data).
	An exception will be thrown if a movement through the left cannot be matched
	in the right (or if the right side has extra fields).
	
	See elemConnect for details on how the root connections are issued.
	


	.. chisel:attr:: def connect(sourceInfo: SourceInfo, connectCompileOptions: CompileOptions, left: Data, right: Data, context_mod: RawModule): Unit =

	
		This function is what recursively tries to connect a left and right together	
		There is some cleverness in the use of internal try-catch to catch exceptions
		during the recursive decent and then rethrow them with extra information added.
		This gives the user a 'path' to where in the connections things went wrong.
	  


