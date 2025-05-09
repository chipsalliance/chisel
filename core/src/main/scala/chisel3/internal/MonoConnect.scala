// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import _root_.firrtl.ir.ClassPropertyType
import chisel3._
import chisel3.experimental.{Analog, BaseModule, SourceInfo}
import chisel3.internal.binding._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir.{Block, Connect, DefInvalid, ProbeDefine, PropAssign}
import chisel3.internal.firrtl.Converter
import chisel3.experimental.dataview.{isView, reify, reifyIdentityView}
import chisel3.properties.{Class, Property}
import chisel3.reflect.DataMirror

import scala.annotation.tailrec

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

private[chisel3] object MonoConnect {
  def formatName(data: Data) = s"""${data.earlyName} in ${data.parentNameOpt.getOrElse("(unknown)")}"""

  // These are all the possible exceptions that can be thrown.
  // These are from element-level connection
  def UnreadableSourceException(sink: Data, source: Data) =
    MonoConnectException(
      s"""${formatName(source)} cannot be read from module ${sink.parentNameOpt.getOrElse("(unknown)")}."""
    )
  def UnwritableSinkException(sink: Data, source: Data) =
    MonoConnectException(
      s"""${formatName(sink)} cannot be written from module ${source.parentNameOpt.getOrElse("(unknown)")}."""
    )
  def escapedScopeErrorMsg(data: Data, blockInfo: SourceInfo) = {
    s"operand '$data' has escaped the scope of the block (${blockInfo.makeMessage()}) in which it was constructed."
  }
  def UnknownRelationException =
    MonoConnectException("Sink or source unavailable to current module.")
  // These are when recursing down aggregate types
  def MismatchedVecException =
    MonoConnectException("Sink and Source are different length Vecs.")
  def MissingFieldException(field: String) =
    MonoConnectException(s"Source Record missing field ($field).")
  def MismatchedException(sink: Data, source: Data) =
    MonoConnectException(
      s"Sink (${sink.cloneType.toString}) and Source (${source.cloneType.toString}) have different types."
    )
  def DontCareCantBeSink =
    MonoConnectException("DontCare cannot be a connection sink")
  def AnalogCantBeMonoSink(sink: Data) =
    MonoConnectException(s"Sink ${formatName(sink)} of type Analog cannot participate in a mono connection (:=)")
  def AnalogCantBeMonoSource(source: Data) =
    MonoConnectException(s"Source ${formatName(source)} of type Analog cannot participate in a mono connection (:=)")
  def AnalogMonoConnectionException(source: Data, sink: Data) =
    MonoConnectException(
      s"Source ${formatName(source)} and sink ${formatName(sink)} of type Analog cannot participate in a mono connection (:=)"
    )
  def SourceProbeMonoConnectionException(source: Data) =
    MonoConnectException(s"Source ${formatName(source)} of Probed type cannot participate in a mono connection (:=)")
  def SinkProbeMonoConnectionException(sink: Data) =
    MonoConnectException(s"Sink ${formatName(sink)} of Probed type cannot participate in a mono connection (:=)")

  /** Check if the argument is visible from current block scope
    *
    * Returns source locator of original block declaration if not visible (for error reporting)
    *
    * @return None if visible, Some(location of original block declaration)
    */
  def checkBlockVisibility(x: Data): Option[SourceInfo] = {
    require(Builder.hasDynamicContext, "block visibility cannot currently be determined without context")
    // TODO: Checking block stack isn't quite right in situations where the stack of blocks
    // doesn't reflect the generated structure (for example using withRegion to jump multiple levels).
    def visible(b: Block) = Builder.blockStack.contains(b)
    x.topBinding match {
      case mp: MemoryPortBinding =>
        None // TODO (albert-magyar): remove this "bridge" for odd enable logic of current CHIRRTL memories
      case DynamicIndexBinding(vec) => checkBlockVisibility(vec) // This goes with the MemoryPortBinding hack
      case bb: BlockBinding => bb.parentBlock.collect { case b: Block if !visible(b) => b.sourceInfo }
      case _ => None
    }
  }

  private[chisel3] def reportIfReadOnly[T <: Data](x: (T, ViewWriteability))(implicit info: SourceInfo): T = {
    val (data, writable) = x
    writable.reportIfReadOnly(data)(WireInit(data))
  }

  /** This function is what recursively tries to connect a sink and source together
    *
    * There is some cleverness in the use of internal try-catch to catch exceptions
    * during the recursive decent and then rethrow them with extra information added.
    * This gives the user a 'path' to where in the connections things went wrong.
    */
  def connect(
    sourceInfo:  SourceInfo,
    sink:        Data,
    source:      Data,
    context_mod: BaseModule
  ): Unit = {
    (sink, source) match {
      // Two probes are connected at the root.
      case (sink_e, source_e)
          if (DataMirror.hasProbeTypeModifier(sink_e) && DataMirror.hasProbeTypeModifier(source_e)) =>
        probeDefine(sourceInfo, sink_e, source_e, context_mod)

      // A probe-y thing cannot be connected to a different probe-y thing.
      case (_, source_e: Data) if DataMirror.hasProbeTypeModifier(source_e) =>
        throw SourceProbeMonoConnectionException(source_e)
      case (sink_e: Data, _) if DataMirror.hasProbeTypeModifier(sink_e) =>
        throw SinkProbeMonoConnectionException(sink_e)

      // Handle legal element cases, note (Bool, Bool) is caught by the first two, as Bool is a UInt
      case (sink_e: Bool, source_e: UInt) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: UInt, source_e: Bool) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: UInt, source_e: UInt) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: SInt, source_e: SInt) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: Clock, source_e: Clock) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: AsyncReset, source_e: AsyncReset) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: ResetType, source_e: Reset) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: Reset, source_e: ResetType) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: EnumType, source_e: UnsafeEnum) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: EnumType, source_e: EnumType) if sink_e.typeEquivalent(source_e) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_e: UnsafeEnum, source_e: UInt) =>
        elemConnect(sourceInfo, sink_e, source_e, context_mod)
      case (sink_p: Property[_], source_p: Property[_]) =>
        propConnect(sourceInfo, sink_p, source_p, context_mod)

      // Handle Vec case
      case (sink_v: Vec[Data @unchecked], source_v: Vec[Data @unchecked]) =>
        if (sink_v.length != source_v.length) { throw MismatchedVecException }

        val sinkReified: Option[Vec[Data @unchecked]] =
          if (isView(sink_v)) reifyIdentityView(sink_v).map(reportIfReadOnly(_)(sourceInfo)) else Some(sink_v)
        val sourceReified: Option[Vec[Data @unchecked]] =
          if (isView(source_v)) reifyIdentityView(source_v).map(_._1) else Some(source_v)

        if (
          sinkReified.nonEmpty && sourceReified.nonEmpty && canFirrtlConnectData(
            sinkReified.get,
            sourceReified.get,
            sourceInfo,
            context_mod
          )
        ) {
          pushCommand(Connect(sourceInfo, sinkReified.get.lref(sourceInfo), sourceReified.get.ref(sourceInfo)))
        } else {
          for (idx <- 0 until sink_v.length) {
            try {
              connect(sourceInfo, sink_v(idx), source_v(idx), context_mod)
            } catch {
              case MonoConnectException(message) => throw MonoConnectException(s"($idx)$message")
            }
          }
        }
      // Handle Vec connected to DontCare. Apply the DontCare to individual elements.
      case (sink_v: Vec[Data @unchecked], DontCare) =>
        for (idx <- 0 until sink_v.length) {
          try {
            connect(sourceInfo, sink_v(idx), source, context_mod)
          } catch {
            case MonoConnectException(message) => throw MonoConnectException(s"($idx)$message")
          }
        }

      // Handle Record case
      case (sink_r: Record, source_r: Record) =>
        val sinkReified: Option[Record] =
          if (isView(sink_r)) reifyIdentityView(sink_r).map(reportIfReadOnly(_)(sourceInfo)) else Some(sink_r)
        val sourceReified: Option[Record] =
          if (isView(source_r)) reifyIdentityView(source_r).map(_._1) else Some(source_r)

        if (
          sinkReified.nonEmpty && sourceReified.nonEmpty && canFirrtlConnectData(
            sinkReified.get,
            sourceReified.get,
            sourceInfo,
            context_mod
          )
        ) {
          pushCommand(Connect(sourceInfo, sinkReified.get.lref(sourceInfo), sourceReified.get.ref(sourceInfo)))
        } else {
          // For each field, descend with right
          for ((field, sink_sub) <- sink_r._elements) {
            try {
              source_r._elements.get(field) match {
                case Some(source_sub) => connect(sourceInfo, sink_sub, source_sub, context_mod)
                case None             => throw MissingFieldException(field)
              }
            } catch {
              case MonoConnectException(message) => throw MonoConnectException(s".$field$message")
            }
          }
        }
      // Handle Record connected to DontCare. Apply the DontCare to individual elements.
      case (sink_r: Record, DontCare) =>
        // For each field, descend with right
        for ((field, sink_sub) <- sink_r._elements) {
          try {
            connect(sourceInfo, sink_sub, source, context_mod)
          } catch {
            case MonoConnectException(message) => throw MonoConnectException(s".$field$message")
          }
        }

      // Source is DontCare - it may be connected to anything. It generates a defInvalid for the sink.
      case (_sink: Element, DontCare) =>
        val (sink, writable) = reify(_sink) // Handle views.
        writable.reportIfReadOnlyUnit {
          pushCommand(DefInvalid(sourceInfo, sink.lref(sourceInfo))): Unit
        }(sourceInfo) // Nothing to push if an error.

      // DontCare as a sink is illegal.
      case (DontCare, _) => throw DontCareCantBeSink
      // Analog is illegal in mono connections.
      case (_: Analog, _: Analog) => throw AnalogMonoConnectionException(source, sink)
      // Analog is illegal in mono connections.
      case (_: Analog, _) => throw AnalogCantBeMonoSink(sink)
      // Analog is illegal in mono connections.
      case (_, _: Analog) => throw AnalogCantBeMonoSource(source)
      // Sink and source are different subtypes of data so fail
      case (sink, source) => throw MismatchedException(sink, source)
    }
  }

  /** Determine if a valid connection can be made between a source [[Data]] and sink
    * [[Data]] given their parent module and directionality context
    *
    * @return whether the source and sink exist in an appropriate context to be connected
    */
  private[chisel3] def dataConnectContextCheck(
    implicit sourceInfo: SourceInfo,
    sink:                Data,
    source:              Data,
    context_mod:         BaseModule
  ): Boolean = {
    import ActualDirection.{Bidirectional, Input, Output}
    // If source has no location, assume in context module
    // This can occur if is a literal, unbound will error previously
    val sink_mod:   BaseModule = sink.topBinding.location.getOrElse(throw UnwritableSinkException(sink, source))
    val source_mod: BaseModule = source.topBinding.location.getOrElse(context_mod)

    val sink_parent_opt = Builder.retrieveParent(sink_mod, context_mod)
    val source_parent_opt = Builder.retrieveParent(source_mod, context_mod)
    val context_mod_opt = Some(context_mod)

    val sink_is_port = sink.topBinding match {
      case PortBinding(_) => true
      case _              => false
    }
    val source_is_port = source.topBinding match {
      case PortBinding(_) => true
      case _              => false
    }

    checkBlockVisibility(sink) match {
      case Some(blockInfo) => Builder.error(escapedScopeErrorMsg(sink, blockInfo))
      case None            => ()
    }

    checkBlockVisibility(source) match {
      case Some(blockInfo) => Builder.error(escapedScopeErrorMsg(source, blockInfo))
      case None            => ()
    }

    // CASE: Context is same module that both sink node and source node are in
    if ((context_mod == sink_mod) && (context_mod == source_mod)) {
      !sink_is_port || sink.direction != Input
    }

    // CASE: Context is same module as sink node and source node is in a child module
    else if ((sink_mod == context_mod) && (source_parent_opt == context_mod_opt)) {
      // NOTE: Workaround for bulk connecting non-agnostified FIRRTL ports
      // See: https://github.com/freechipsproject/firrtl/issues/1703
      // Original behavior should just check if the sink direction is an Input
      val sinkCanBeInput = sink.direction match {
        case Input => true
        case _: Bidirectional => true
        case _ => false
      }
      // Thus, right node better be a port node and thus have a direction
      if (!source_is_port) { false }
      else if (sinkCanBeInput) {
        if (source.direction == Output) { true }
        else { false }
      } else { true }
    }

    // CASE: Context is same module as source node and sink node is in child module
    else if ((source_mod == context_mod) && (sink_parent_opt == context_mod_opt)) {
      // NOTE: Workaround for bulk connecting non-agnostified FIRRTL ports
      // See: https://github.com/freechipsproject/firrtl/issues/1703
      // Original behavior should just check if the sink direction is an Input
      sink.direction match {
        case Input => true
        case _: Bidirectional => true
        case _ => false
      }
    }

    // CASE: Context is the parent module of both the module containing sink node
    //                                        and the module containing source node
    //   Note: This includes case when sink and source in same module but in parent
    else if ((sink_parent_opt == context_mod_opt) && (source_parent_opt == context_mod_opt)) {
      // Thus both nodes must be ports and have a direction
      if (!source_is_port) { false }
      else if (sink_is_port) {
        // NOTE: Workaround for bulk connecting non-agnostified FIRRTL ports
        // See: https://github.com/freechipsproject/firrtl/issues/1703
        // Original behavior should just check if the sink direction is an Input
        sink.direction match {
          case Input => true
          case _: Bidirectional => true // NOTE: Workaround for non-agnostified ports
          case _ => false
        }
      } else { false }
    }

    // Not quite sure where left and right are compared to current module
    // so just error out
    else false
  }

  /** Trace flow from child Data to its parent.
    *
    * Returns true if, given the context,
    * this signal can be a sink when wantsToBeSink = true,
    * or if it can be a source when wantsToBeSink = false.
    * Always returns true if the Data does not actually correspond
    * to a Port.
    */
  @tailrec private[chisel3] def traceFlow(
    wantToBeSink:     Boolean,
    currentlyFlipped: Boolean,
    data:             Data,
    context_mod:      BaseModule
  ): Boolean = {
    val sdir = data.specifiedDirection
    val coercedFlip = sdir == SpecifiedDirection.Input
    val coercedAlign = sdir == SpecifiedDirection.Output
    val flipped = sdir == SpecifiedDirection.Flip
    val traceFlipped = ((flipped ^ currentlyFlipped) || coercedFlip) && (!coercedAlign)
    data.binding.get match {
      case ChildBinding(parent) => traceFlow(wantToBeSink, traceFlipped, parent, context_mod)
      case PortBinding(enclosure) =>
        val childPort = enclosure != context_mod
        wantToBeSink ^ childPort ^ traceFlipped
      case _ => true
    }
  }
  def canBeSink(data:   Data, context_mod: BaseModule): Boolean = traceFlow(true, false, data, context_mod)
  def canBeSource(data: Data, context_mod: BaseModule): Boolean = traceFlow(false, false, data, context_mod)

  /** Check whether two Data can be bulk connected (<=) in FIRRTL. (MonoConnect case)
    *
    * Mono-directional bulk connects only work if all signals of the sink are unidirectional
    * In the case of a sink aggregate with bidirectional signals, e.g. `Decoupled`,
    * a `BiConnect` (`chisel3.<>` or `chisel.:<>=`) is necessary.
    */
  private[chisel3] def canFirrtlConnectData(
    sink:        Data,
    source:      Data,
    sourceInfo:  SourceInfo,
    context_mod: BaseModule
  ): Boolean = {
    // Assuming we're using a <>, check if a FIRRTL.<= connection operator is valid in that case
    def biConnectCheck =
      BiConnect.canFirrtlConnectData(sink, source, sourceInfo, context_mod)

    // Check that the sink Data can be driven (not bidirectional or an input) to match Chisel semantics
    def sinkCanBeDrivenCheck: Boolean =
      sink.direction == ActualDirection.Output || sink.direction == ActualDirection.Unspecified

    biConnectCheck && sinkCanBeDrivenCheck
  }

  // This function (finally) issues the connection operation
  private def issueConnect(sink: Element, source: Element)(implicit sourceInfo: SourceInfo): Unit = {
    // If the source is a DontCare, generate a DefInvalid for the sink,
    //  otherwise, issue a Connect.
    source.topBinding match {
      case b: DontCareBinding => pushCommand(DefInvalid(sourceInfo, sink.lref))
      case _ => pushCommand(Connect(sourceInfo, sink.lref, source.ref))
    }
  }

  // This function checks if element-level connection operation allowed.
  // Then it either issues it or throws the appropriate exception.
  def elemConnect(
    implicit sourceInfo: SourceInfo,
    _sink:               Element,
    _source:             Element,
    context_mod:         BaseModule
  ): Unit = {
    // Reify sink and source if they're views.
    val (sink, writable) = reify(_sink)
    val (source, _) = reify(_source)

    checkConnect(sourceInfo, sink, source, context_mod)
    writable.reportIfReadOnlyUnit(issueConnect(sink, source))
  }

  def propConnect(
    sourceInfo: SourceInfo,
    sinkProp:   Property[_],
    sourceProp: Property[_],
    context:    BaseModule
  ): Unit = {
    implicit val info: SourceInfo = sourceInfo
    // Reify sink and source if they're views.
    val (sink, writable) = reify(sinkProp)
    val (source, _) = reify(sourceProp)

    checkConnect(sourceInfo, sink, source, context)
    // Add the PropAssign command directly onto the correct BaseModule subclass.
    context match {
      case rm: RawModule =>
        writable.reportIfReadOnlyUnit {
          rm.addCommand(PropAssign(sourceInfo, sink.lref, source.ref))
        }(sourceInfo)
      case cls: Class =>
        writable.reportIfReadOnlyUnit {
          cls.addCommand(PropAssign(sourceInfo, sink.lref, source.ref))
        }(sourceInfo)
      case _ => throwException("Internal Error! Property connection can only occur within RawModule or Class.")
    }
  }

  def probeDefine(
    sourceInfo:  SourceInfo,
    sinkProbe:   Data,
    sourceProbe: Data,
    context:     BaseModule
  ): Unit = {
    implicit val info: SourceInfo = sourceInfo
    val (sink, writable) = reifyIdentityView(sinkProbe).getOrElse(
      throwException(
        s"If a DataView contains a Probe, it must resolve to one Data. $sinkProbe does not meet this criteria."
      )
    )
    val (source, _) = reifyIdentityView(sourceProbe).getOrElse(
      throwException(
        s"If a DataView contains a Probe, it must resolve to one Data. $sourceProbe does not meet this criteria."
      )
    )
    checkConnect.checkConnection(sourceInfo, sink, source, context)
    context match {
      case rm: RawModule =>
        writable.reportIfReadOnlyUnit {
          rm.addCommand(ProbeDefine(sourceInfo, sink.lref, source.ref))
        }(sourceInfo) // Nothing to push if an error.
      case _ => throwException("Internal Error! Probe connection can only occur within RawModule.")
    }
  }
}

/** This object can be applied to check if element-level connection is allowed.
  *
  * Its apply methods throw the appropriate exception, if necessary.
  */
private[chisel3] object checkConnect {
  def apply(
    sourceInfo:  SourceInfo,
    sink:        Element,
    source:      Element,
    context_mod: BaseModule
  ): Unit = {
    checkConnection(sourceInfo, sink, source, context_mod)

    // Extra checks for Property[_] elements.
    (sink, source) match {
      case (sinkProp: Property[_], sourceProp: Property[_]) =>
        checkConnect.checkPropertyConnection(sourceInfo, sinkProp, sourceProp)
      case (_, _) =>
    }
  }

  def checkConnection(
    sourceInfo:  SourceInfo,
    sink:        Data,
    source:      Data,
    context_mod: BaseModule
  ): Unit = {
    import BindingDirection.{Input, Internal, Output} // Using extensively so import these

    // Import helpers and exception types.
    import MonoConnect.{
      checkBlockVisibility,
      escapedScopeErrorMsg,
      UnknownRelationException,
      UnreadableSourceException,
      UnwritableSinkException
    }

    // If source has no location, assume in context module
    // This can occur if is a literal, unbound will error previously
    val sink_mod:   BaseModule = sink.topBinding.location.getOrElse(throw UnwritableSinkException(sink, source))
    val source_mod: BaseModule = source.topBinding.location.getOrElse(context_mod)

    val sink_parent_opt = Builder.retrieveParent(sink_mod, context_mod)
    val source_parent_opt = Builder.retrieveParent(source_mod, context_mod)
    val context_mod_opt = Some(context_mod)

    val sink_direction = BindingDirection.from(sink.topBinding, sink.direction)
    val source_direction = BindingDirection.from(source.topBinding, source.direction)

    // CASE: Context is same module that both left node and right node are in
    if ((context_mod == sink_mod) && (context_mod == source_mod)) {
      ((sink_direction, source_direction): @unchecked) match {
        //    SINK          SOURCE
        //    CURRENT MOD   CURRENT MOD
        case (Output, _)   => ()
        case (Internal, _) => ()
        case (Input, _)    => throw UnwritableSinkException(sink, source)
      }
    }

    // CASE: Context is same module as sink node and right node is in a child module
    else if ((sink_mod == context_mod) && (source_parent_opt == context_mod_opt)) {
      // Thus, right node better be a port node and thus have a direction
      ((sink_direction, source_direction): @unchecked) match {
        //    SINK          SOURCE
        //    CURRENT MOD   CHILD MOD
        case (Internal, Output) => ()
        case (Internal, Input)  => ()
        case (Output, Output)   => ()
        case (Output, Input)    => ()
        case (_, Internal)      => throw UnreadableSourceException(sink, source)
        case (Input, Output)    => ()
        case (Input, _)         => throw UnwritableSinkException(sink, source)
      }
    }

    // CASE: Context is same module as source node and sink node is in child module
    else if ((source_mod == context_mod) && (sink_parent_opt == context_mod_opt)) {
      // Thus, left node better be a port node and thus have a direction
      ((sink_direction, source_direction): @unchecked) match {
        //    SINK          SOURCE
        //    CHILD MOD     CURRENT MOD
        case (Input, _)    => ()
        case (Output, _)   => throw UnwritableSinkException(sink, source)
        case (Internal, _) => throw UnwritableSinkException(sink, source)
      }
    }

    // CASE: Context is the parent module of both the module containing sink node
    //                                        and the module containing source node
    //   Note: This includes case when sink and source in same module but in parent
    else if ((sink_parent_opt == context_mod_opt) && (source_parent_opt == context_mod_opt)) {
      // Thus both nodes must be ports and have a direction
      ((sink_direction, source_direction): @unchecked) match {
        //    SINK          SOURCE
        //    CHILD MOD     CHILD MOD
        case (Input, Input)  => ()
        case (Input, Output) => ()
        case (Output, _)     => throw UnwritableSinkException(sink, source)
        case (_, Internal)   => throw UnreadableSourceException(sink, source)
        case (Internal, _)   => throw UnwritableSinkException(sink, source)
      }
    }

    // Not quite sure where left and right are compared to current module
    // so just error out
    else throw UnknownRelationException

    checkBlockVisibility(sink) match {
      case Some(blockInfo) => Builder.error(escapedScopeErrorMsg(sink, blockInfo))(sourceInfo)
      case None            => ()
    }

    checkBlockVisibility(source) match {
      case Some(blockInfo) => Builder.error(escapedScopeErrorMsg(source, blockInfo))(sourceInfo)
      case None            => ()
    }
  }

  // Extra checks specific to Property types. Most of the checks are baked into the Scala type system, but the actual
  // class names underlying a Property[ClassType] are runtime values, so the check that they match is a runtime check.
  def checkPropertyConnection(sourceInfo: SourceInfo, sink: Property[_], source: Property[_]): Unit = {
    // If source is a ClassType, get the source Property[ClassType] expected class name.
    val sourceClassNameOpt = source.getPropertyType match {
      case ClassPropertyType(name) => Some(name)
      case _                       => None
    }

    // If sink is a ClassType, get the sink Property[ClassType] expected class name.
    val sinkClassNameOpt = sink.getPropertyType match {
      case ClassPropertyType(name) => Some(name)
      case _                       => None
    }

    // We are guaranteed that source and sink are the same Property type by the Scala type system, but if they're both
    // Property[ClassType] we need to check they are the same class.
    (sourceClassNameOpt, sinkClassNameOpt) match {
      case (Some(sourceClassName), Some(sinkClassName)) if (sinkClassName != sourceClassName) =>
        throwException(
          sourceInfo.makeMessage(info =>
            s"Sink Property[ClassType] expected class $sinkClassName, but source Instance[Class] was class $sourceClassName $info"
          )
        )
      case (_, _) => ()
    }
  }
}
