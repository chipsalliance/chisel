// See LICENSE for license details.

package chisel3.core

import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.Connect
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DeprecatedSourceInfo, SourceInfo, SourceInfoTransform, UnlocatableSourceInfo, WireTransform}

/**
* MonoConnect.connect executes a mono-directional connection element-wise.
*
* Note that this isn't commutative. There is an explicit source and sink
* already determined before this function is called.
*
* The connect operation will recurse down the left Data (with the right Data).
* An exception will be thrown if a movement through the left cannot be matched
* in the right. The right side is allowed to have extra Record fields.
* Vecs must still be exactly the same size.
*
* See elemConnect for details on how the root connections are issued.
*
* Note that a valid sink must be writable so, one of these must hold:
* - Is an internal writable node (Reg or Wire)
* - Is an output of the current module
* - Is an input of a submodule of the current module
*
* Note that a valid source must be readable so, one of these must hold:
* - Is an internal readable node (Reg, Wire, Op)
* - Is a literal
* - Is a port of the current module or submodule of the current module
*/

object MonoConnect {
  // These are all the possible exceptions that can be thrown.
  case class MonoConnectException(message: String) extends Exception(message)
  // These are from element-level connection
  def UnreadableSourceException =
    MonoConnectException(": Source is unreadable from current module.")
  def UnwritableSinkException =
    MonoConnectException(": Sink is unwriteable by current module.")
  def UnknownRelationException =
    MonoConnectException(": Sink or source unavailable to current module.")
  // These are when recursing down aggregate types
  def MismatchedVecException =
    MonoConnectException(": Sink and Source are different length Vecs.")
  def MissingFieldException(field: String) =
    MonoConnectException(s": Source Record missing field ($field).")
  def MismatchedException(sink: String, source: String) =
    MonoConnectException(s": Sink ($sink) and Source ($source) have different types.")

  /** This function is what recursively tries to connect a sink and source together
  *
  * There is some cleverness in the use of internal try-catch to catch exceptions
  * during the recursive decent and then rethrow them with extra information added.
  * This gives the user a 'path' to where in the connections things went wrong.
  */
  def connect(sourceInfo: SourceInfo, connectCompileOptions: CompileOptions, sink: Data, source: Data, context_mod: Module): Unit =
    (sink, source) match {
      // Handle element case (root case)
      case (sink_e: Element, source_e: Element) => {
        elemConnect(sourceInfo, connectCompileOptions, sink_e, source_e, context_mod)
        // TODO(twigg): Verify the element-level classes are connectable
      }
      // Handle Vec case
      case (sink_v: Vec[Data @unchecked], source_v: Vec[Data @unchecked]) => {
        if(sink_v.length != source_v.length) { throw MismatchedVecException }
        for(idx <- 0 until sink_v.length) {
          try {
            connect(sourceInfo, connectCompileOptions, sink_v(idx), source_v(idx), context_mod)
          } catch {
            case MonoConnectException(message) => throw MonoConnectException(s"($idx)$message")
          }
        }
      }
      // Handle Record case
      case (sink_r: Record, source_r: Record) => {
        // For each field, descend with right
        for((field, sink_sub) <- sink_r.elements) {
          try {
            source_r.elements.get(field) match {
              case Some(source_sub) => connect(sourceInfo, connectCompileOptions, sink_sub, source_sub, context_mod)
              case None => {
                if (connectCompileOptions.connectFieldsMustMatch) {
                  throw MissingFieldException(field)
                }
              }
            }
          } catch {
            case MonoConnectException(message) => throw MonoConnectException(s".$field$message")
          }
        }
      }
      // Sink and source are different subtypes of data so fail
      case (sink, source) => throw MismatchedException(sink.toString, source.toString)
    }

  // This function (finally) issues the connection operation
  private def issueConnect(sink: Element, source: Element)(implicit sourceInfo: SourceInfo): Unit = {
    pushCommand(Connect(sourceInfo, sink.lref, source.ref))
  }

  // This function checks if element-level connection operation allowed.
  // Then it either issues it or throws the appropriate exception.
  def elemConnect(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions, sink: Element, source: Element, context_mod: Module): Unit = {
    import Direction.{Input, Output} // Using extensively so import these
    // If source has no location, assume in context module
    // This can occur if is a literal, unbound will error previously
    val sink_mod: Module   = sink.binding.location.getOrElse(throw UnwritableSinkException)
    val source_mod: Module = source.binding.location.getOrElse(context_mod)

    val sink_direction: Option[Direction] = sink.binding.direction
    val source_direction: Option[Direction] = source.binding.direction
    // None means internal

    // CASE: Context is same module that both left node and right node are in
    if( (context_mod == sink_mod) && (context_mod == source_mod) ) {
      ((sink_direction, source_direction): @unchecked) match {
        //    SINK          SOURCE
        //    CURRENT MOD   CURRENT MOD
        case (Some(Output), _) => issueConnect(sink, source)
        case (None,         _) => issueConnect(sink, source)
        case (Some(Input),  _) => throw UnwritableSinkException
      }
    }

    // CASE: Context is same module as sink node and right node is in a child module
    else if( (sink_mod == context_mod) &&
             (source_mod._parent.map(_ == context_mod).getOrElse(false)) ) {
      // Thus, right node better be a port node and thus have a direction
      ((sink_direction, source_direction): @unchecked) match {
        //    SINK          SOURCE
        //    CURRENT MOD   CHILD MOD
        case (None,         Some(Output)) => issueConnect(sink, source)
        case (None,         Some(Input))  => issueConnect(sink, source)
        case (Some(Output), Some(Output)) => issueConnect(sink, source)
        case (Some(Output), Some(Input))  => issueConnect(sink, source)
        case (_,            None) => {
          if (!(connectCompileOptions.dontAssumeDirectionality)) {
            issueConnect(sink, source)
          } else {
            throw UnreadableSourceException
          }
        }
        case (Some(Input),  Some(Output)) if (!(connectCompileOptions.dontTryConnectionsSwapped)) => issueConnect(source, sink)
        case (Some(Input),  _)    => throw UnwritableSinkException
      }
    }

    // CASE: Context is same module as source node and sink node is in child module
    else if( (source_mod == context_mod) &&
             (sink_mod._parent.map(_ == context_mod).getOrElse(false)) ) {
      // Thus, left node better be a port node and thus have a direction
      ((sink_direction, source_direction): @unchecked) match {
        //    SINK          SOURCE
        //    CHILD MOD     CURRENT MOD
        case (Some(Input),  _) => issueConnect(sink, source)
        case (Some(Output), _) => throw UnwritableSinkException
        case (None,         _) => throw UnwritableSinkException
      }
    }

    // CASE: Context is the parent module of both the module containing sink node
    //                                        and the module containing source node
    //   Note: This includes case when sink and source in same module but in parent
    else if( (sink_mod._parent.map(_ == context_mod).getOrElse(false)) &&
             (source_mod._parent.map(_ == context_mod).getOrElse(false))
    ) {
      // Thus both nodes must be ports and have a direction
      ((sink_direction, source_direction): @unchecked) match {
        //    SINK          SOURCE
        //    CHILD MOD     CHILD MOD
        case (Some(Input),  Some(Input))  => issueConnect(sink, source)
        case (Some(Input),  Some(Output)) => issueConnect(sink, source)
        case (Some(Output), _)            => throw UnwritableSinkException
        case (_,            None) => {
          if (!(connectCompileOptions.dontAssumeDirectionality)) {
            issueConnect(sink, source)
          } else {
            throw UnreadableSourceException
          }
        }
        case (None,         _)            => throw UnwritableSinkException
      }
    }

    // Not quite sure where left and right are compared to current module
    // so just error out
    else throw UnknownRelationException
  }
}
