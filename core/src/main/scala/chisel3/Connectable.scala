// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.{prefix, BiConnect, Builder}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.{Analog, DataMirror}
import chisel3.experimental.Defaulting._

import scala.collection.mutable
import chisel3.internal.ChildBinding

/** The default connection operators for Chisel hardware components */
object Connectable {

  /** ConnectableData Typeclass defines the following operators on all subclasses of Data: :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableData[T <: Data](consumer: T) {

    /** The "aligned connection operator" between a producer and consumer.
      *
      * For `consumer :<= producer`, each of consumer's leaf fields WHO ARE ALIGNED WITH RESPECT TO CONSUMER are driven from the corresponding producer leaf field
      * All producer's leaf/branch alignments (with respect to producer) do not influence the connection.
      *
      * The following restrictions apply:
      *  - The Chisel type of consumer and producer must be the "same shape" recursively:
      *    - All ground types are the same (UInt and UInt are same, SInt and UInt are not), but widths can be different
      *    - All vector types are the same length
      *    - All bundle types have the same field names, but the flips of fields can be different between producer and consumer
      *  - The leaf fields that are ultimately assigned to, must be assignable. This means they cannot be module inputs or instance outputs.
      *
      * @note Connecting two [[Decoupled]]'s would connect `bits` and `valid` from producer to consumer, but leave `ready` unconnected
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connection ("aligned connection")
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      * @param sourceInfo
      * @group connection
      */
    final def :<=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessEq)
      }
    }

    /** The "flipped connection operator" between a producer and consumer.
      *
      * For `consumer :>= producer`, each of producers's leaf fields WHO ARE FLIPPED WITH RESPECT TO PRODUCER are driven from the corresponding consumer leaf field
      * All consumer's leaf/branch alignments (with respect to consumer) do not influence the connection.
      *
      * The following restrictions apply:
      *  - The Chisel type of consumer and producer must be the "same shape":
      *    - All ground types are the same (UInt and UInt are same, SInt and UInt are not), but widths can be different
      *    - All vector types are the same length
      *    - All bundle types have the same field names, but the flips of fields can be different
      *  - The leaf fields that are ultimately assigned to, must be assignable. This means they cannot be module inputs or instance outputs.
      *
      * @note Connecting two [[Decoupled]]'s would connect `ready` from consumer to producer, but leave `bits` and `valid` unconnected
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("flipped connection")
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      * @param sourceInfo
      * @group connection
      */
    final def :>=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonGreaterEq)
      }
    }

    /** The "bi-direction connection operator", aka the "tur-duck-en operator"
      *
      * For `consumer :<>= producer`, both producer and consumer leafs could be driving or be driven-to:
      *   - consumer's fields aligned w.r.t. consumer will be driven by corresponding fields of producer
      *   - producer's fields flipped w.r.t. producer will be driven by corresponding fields of consumer
      *
      * Identical to calling both :<= and :>= in sequence (order is irrelevant), e.g.:
      *   consumer :<= producer
      *   consumer :>= producer
      *
      * @note Connecting two [[Decoupled]]'s would connect `bits` and `valid` from producer to consumer, and `ready` from consumer to producer.
      * @note This may have surprising-to-new-users behavior if the flips of consumer and producer do not match. Save yourself the headache and internalize what
      * :<= and :>= do, and then you'll be able to reason your way to understanding what's happening :)
      * @note If the types of consumer and producer also have identical relative flips, then we can emit FIRRTL.<= as it is a stricter version of chisel3.:<>=
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      * @note "turk-duck-en" is a meme where a turkey is stuffed with a duck, which is stuffed with a chicken; `:<>=` is a `:=` stuffed with a `<>`
      *
      * @param consumer the left-hand-side of the connection (read above comment for more info)
      * @param producer the right-hand-side of the connection (read above comment for more info)
      * @param sourceInfo
      * @group connection
      */
    final def :<>=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        val canFirrtlConnect =
          try {
            BiConnect.canFirrtlConnectData(
              consumer,
              producer,
              sourceInfo,
              DirectionalConnectionFunctions.compileOptions,
              Builder.referenceUserModule
            )
          } catch {
            // For some reason, an error is thrown if its a View; since this is purely an optimization, any actual error would get thrown
            //  when calling DirectionConnectionFunctions.assign. Hence, we can just default to false to take the non-optimized emission path
            case e: Throwable => false
          }
        if (canFirrtlConnect) {
          consumer.firrtlConnect(producer)
        } else {
          // cannot call :<= and :>= directly because otherwise prefix is called twice
          DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessGreaterEq)
        }
      }
    }

    /** The "mono-direction connection operator", aka the "coercion operator"
      *
      * For `consumer :#= producer`, all leaf fields of consumer (regardless of relative flip) are driven by the corresponding leaf fields of producer (regardless of relative flip)
      *
      * Identical to calling :<= and :>=, but swapping consumer/producer for :>=: (order is irrelevant), e.g.:
      *   consumer :<= producer
      *   producer :>= consumer
      *
      * @note Connecting two [[Decoupled]]'s would connect `bits`, `valid`, AND `ready` from producer to consumer (despite `ready` being flipped)
      * @note Functionally equivalent to chisel3.:=, but different than Chisel.:=
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection, all fields will be driven-to
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      * @param sourceInfo
      * @group connection
      */
    final def :#=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonHashEq)
      }
    }
  }

  /** ConnectableVec Typeclass defines the following operators on between a (consumer: Vec) and (producer: Seq): :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableVec[T <: Data](consumer: Vec[T]) {

    /** The "aligned connection operator" between a producer and consumer.
      *
      * For `consumer :<= producer`, each of consumer's leaf fields WHO ARE ALIGNED WITH RESPECT TO CONSUMER are driven from the corresponding producer leaf field
      * All producer's leaf/branch alignments (with respect to producer) do not influence the connection.
      *
      * The following restrictions apply:
      *  - The Chisel type of consumer and producer must be the "same shape" recursively:
      *    - All ground types are the same (UInt and UInt are same, SInt and UInt are not), but widths can be different
      *    - All vector types are the same length
      *    - All bundle types have the same field names, but the flips of fields can be different between producer and consumer
      *  - The leaf fields that are ultimately assigned to, must be assignable. This means they cannot be module inputs or instance outputs.
      *
      * @note Connecting two [[Decoupled]]'s would connect `bits` and `valid` from producer to consumer, but leave `ready` unconnected
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connection ("aligned connection")
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      * @param sourceInfo
      * @group connection
      */
    def :<=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :<= b }
    }

    /** The "flipped connection operator" between a producer and consumer.
      *
      * For `consumer :>= producer`, each of producers's leaf fields WHO ARE FLIPPED WITH RESPECT TO PRODUCER are driven from the corresponding consumer leaf field
      * All consumer's leaf/branch alignments (with respect to consumer) do not influence the connection.
      *
      * The following restrictions apply:
      *  - The Chisel type of consumer and producer must be the "same shape":
      *    - All ground types are the same (UInt and UInt are same, SInt and UInt are not), but widths can be different
      *    - All vector types are the same length
      *    - All bundle types have the same field names, but the flips of fields can be different
      *  - The leaf fields that are ultimately assigned to, must be assignable. This means they cannot be module inputs or instance outputs.
      *
      * @note Connecting two [[Decoupled]]'s would connect `ready` from consumer to producer, but leave `bits` and `valid` unconnected
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("flipped connection")
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      * @param sourceInfo
      * @group connection
      */
    def :>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :>= b }
    }

    /** The "bi-direction connection operator", aka the "tur-duck-en operator"
      *
      * For `consumer :<>= producer`, both producer and consumer leafs could be driving or be driven-to:
      *   - consumer's fields aligned w.r.t. consumer will be driven by corresponding fields of producer
      *   - producer's fields flipped w.r.t. producer will be driven by corresponding fields of consumer
      *
      * Identical to calling both :<= and :>= in sequence (order is irrelevant), e.g.:
      *   consumer :<= producer
      *   consumer :>= producer
      *
      * @note Connecting two [[Decoupled]]'s would connect `bits` and `valid` from producer to consumer, and `ready` from consumer to producer.
      * @note This may have surprising-to-new-users behavior if the flips of consumer and producer do not match. Save yourself the headache and internalize what
      * :<= and :>= do, and then you'll be able to reason your way to understanding what's happening :)
      * @note If the types of consumer and producer also have identical relative flips, then we can emit FIRRTL.<= as it is a stricter version of chisel3.:<>=
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection (read above comment for more info)
      * @param producer the right-hand-side of the connection (read above comment for more info)
      * @param sourceInfo
      * @group connection
      */
    def :<>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :<>= b }
    }

    /** The "mono-direction connection operator", aka the "coercion operator"
      *
      * For `consumer :#= producer`, all leaf fields of consumer (regardless of relative flip) are driven by the corresponding leaf fields of producer (regardless of relative flip)
      *
      * Identical to calling :<= and :>=, but swapping consumer/producer for :>=: (order is irrelevant), e.g.:
      *   consumer :<= producer
      *   producer :>= consumer
      *
      * @note Connecting two [[Decoupled]]'s would connect `bits`, `valid`, AND `ready` from producer to consumer (despite `ready` being flipped)
      * @note Functionally equivalent to chisel3.:=, but different than Chisel.:=
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection, all fields will be driven-to
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      * @param sourceInfo
      * @group connection
      */
    def :#=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :#= b }
    }
  }
}

private[chisel3] object DirectionalConnectionFunctions {
  // Consumed by the := operator, set to what chisel3 will eventually become.
  implicit val compileOptions = new CompileOptions {

    val connectFieldsMustMatch:      Boolean = true
    val declaredTypeMustBeUnbound:   Boolean = true
    val dontTryConnectionsSwapped:   Boolean = false
    val dontAssumeDirectionality:    Boolean = true
    val checkSynthesizable:          Boolean = true
    val explicitInvalidate:          Boolean = true
    val inferModuleReset:            Boolean = true
    override def emitStrictConnects: Boolean = true
  }

  case class LeafConnection(consumer: Option[(Data, RelativeOrientation)], producer: Option[(Data, RelativeOrientation)])

  def buildTrie(consumer: Data, producer: Data): Trie[String, LeafConnection] = {
    require(consumer != DontCare)

    val trie = Trie.empty[String, LeafConnection]
    val consumerLeafs = getLeafs(Vector.empty[String], consumer, consumer, AlignedWithRoot)
    consumerLeafs.foreach { case (path, (d, o)) =>
      trie.insert(path, LeafConnection(Some((d, o)), None))
    }
    if(producer == DontCare) {
      trie.transform {
        (path, optLeaf) => optLeaf.map {
          case LeafConnection(Some((d, o)), None) => LeafConnection(Some(d, AlignedWithRoot), Some(DontCare, AlignedWithRoot))
        }
      }
    } else {
      val producerLeafs = getLeafs(Vector.empty[String], producer, producer, AlignedWithRoot)
      producerLeafs.foreach { case (path, (d, o)) =>
        trie.get(path) match {
          case None => trie.insert(path, LeafConnection(None, Some((d, o))))
          case Some(LeafConnection(Some((c, co)), None)) =>
            trie.delete(path)
            trie.insert(path, LeafConnection(Some((c, co)), Some((d, o))))
        }
      }
      trie
    }
  }

  def getLeafs(path: Vector[String], d: Data, r: Data, o: RelativeOrientation): Seq[(Vector[String], (Data, RelativeOrientation))] = {
    d match {
      case a: Aggregate if a.getElements.size == 0 => Nil
      case a: Vec[Data @unchecked] =>
        a.getElements.zipWithIndex.flatMap { case (d, i) => getLeafs(path :+ i.toString, d, r, o)}
      case a: Record =>
        a.elements.flatMap { case (field, d) => getLeafs(path :+ field, d, r, deriveOrientation(d, r, o))}.toList
      case x => Seq((path, (x, o)))
    }
  }
  def leafConnect(c: Data, p: Data, o: RelativeOrientation)(implicit sourceInfo: SourceInfo): Unit = {
    require(!c.isInstanceOf[Aggregate] && !p.isInstanceOf[Aggregate])
    (c, p, o) match {
      case (x: Analog, y: Analog, _) => assignAnalog(x, y)
      case (x: Analog, DontCare, _) => assignAnalog(x, DontCare)
      case (x, y, AlignedWithRoot) => c := p
      case (x, y, FlippedWithRoot) => p := c
    }
  }

  def doAssignment(trie: Trie[String, LeafConnection], op: ConnectionOperator)(implicit sourceInfo: SourceInfo): Unit = {
    val errors = mutable.ArrayBuffer[String]()
    trie.collectDeep {
      case (path, Some(LeafConnection(Some((c, co)), Some((p, po))))) if co == po => leafConnect(c, p, co)
      case (path, Some(LeafConnection(Some((c, co)), Some((p, po))))) if co != po =>
        if(op.noWrongOrientations) errors += (s"inversely oriented fields $c and $p") else {
          op match {
            case ColonLessEq => leafConnect(c, p, co)
            case ColonGreaterEq => leafConnect(c, p, po)
            case ColonHashEq => leafConnect(c, p, co)
            case other => throw new Exception("BAD!! Unreachable code is reached, something went wrong")
          }
        }
      case (path, Some(LeafConnection(Some((c, FlippedWithRoot)), None))) => if(op.noDangles) errors += (s"dangling consumer field $c")
      case (path, Some(LeafConnection(None, Some((p, AlignedWithRoot))))) => if(op.noDangles) errors += (s"dangling producer field $p")
      // Defaulting case
      case (path, Some(LeafConnection(Some((c, AlignedWithRoot)), None))) if c.hasDefault => leafConnect(c, c.default, AlignedWithRoot)
      case (path, Some(LeafConnection(None, Some((p, FlippedWithRoot))))) if p.hasDefault => leafConnect(p.default, p, FlippedWithRoot)

      // Non-defaulting case
      case (path, Some(LeafConnection(Some((c, AlignedWithRoot)), None))) => if(op.noUnassigned) errors += (s"unassigned consumer field $c")
      case (path, Some(LeafConnection(None, Some((p, FlippedWithRoot))))) => if(op.noUnassigned) errors += (s"unassigned producer field $p")
      case (path, Some(other)) => throw new Exception("BAD!! Unreachable code is reached, something went wrong")
    }
    if(errors.nonEmpty) {
      Builder.error(errors.mkString("\n"))
    }
  }

  /** Assignment function which implements both :<= and :>=
    *
    * For example, given a connection like so:
    *  c :<= p
    * We can reason the following:
    *  - c is the consumerRoot
    *  - p is the producerRoot
    *  - The '<' indicates that the consumer side (left hand side) is the active side
    *
    * @param consumerRoot the original expression on the left-hand-side of the connection operator
    * @param producerRoot the original expression on the right-hand-side of the connection operator
    * @param activeSide indicates if the connection was a :<= (consumer is active) or :>= (producer is active)
    * @param sourceInfo source info for where the assignment occurred
    */
  def assign(cRoot: Data, pRoot: Data, cOp: ConnectionOperator)(implicit sourceInfo: SourceInfo): Unit = {
    val trie = buildTrie(cRoot, pRoot)
    doAssignment(trie, cOp)
  }


  // Indicates whether the active side is aligned or flipped relative to the active side's root
  sealed trait RelativeOrientation { def invert: RelativeOrientation }
  case object AlignedWithRoot extends RelativeOrientation { def invert = FlippedWithRoot }
  case object FlippedWithRoot extends RelativeOrientation { def invert = AlignedWithRoot }

  sealed trait ConnectionOperator {
    val noDangles: Boolean
    val noUnassigned: Boolean
    val noWrongOrientations: Boolean
  }
  case object ColonLessEq extends ConnectionOperator {
    val noDangles: Boolean = false
    val noUnassigned: Boolean = true
    val noWrongOrientations: Boolean = false
  }
  case object ColonGreaterEq extends ConnectionOperator {
    val noDangles: Boolean = false
    val noUnassigned: Boolean = true
    val noWrongOrientations: Boolean = false
  }
  case object ColonLessGreaterEq extends ConnectionOperator {
    val noDangles: Boolean = true
    val noUnassigned: Boolean = true
    val noWrongOrientations: Boolean = true
  }
  case object ColonHashEq extends ConnectionOperator {
    val noDangles: Boolean = true
    val noUnassigned: Boolean = true
    val noWrongOrientations: Boolean = false
  }

  /** Determines the aligned/flipped of subMember with respect to activeRoot
    *
    * Due to Chisel/chisel3 differences, its a little complicated to calculate the RelativeOrientation, as the information
    *   is captured with both ActualDirection and SpecifiedDirection. Fortunately, all this complexity is captured in this
    *   one function.
    *
    * References activeRoot, defined earlier in the function
    *
    * @param subMember a subfield/subindex of activeRoot (or sub-sub, or sub-sub-sub etc)
    * @param orientation aligned/flipped of d's direct parent aggregate with respect to activeRoot
    * @return orientation aligned/flipped of d with respect to activeRoot
    */
  def deriveOrientation(subMember: Data, root: Data, orientation: RelativeOrientation): RelativeOrientation = {
    (root.direction, subMember.direction, DataMirror.specifiedDirectionOf(subMember)) match {
      case (ActualDirection.Output, ActualDirection.Output, _) => AlignedWithRoot
      case (ActualDirection.Input, ActualDirection.Input, _)   => AlignedWithRoot
      case (ActualDirection.Output, ActualDirection.Input, _)  => FlippedWithRoot
      case (ActualDirection.Input, ActualDirection.Output, _)  => FlippedWithRoot
      case (_, _, SpecifiedDirection.Unspecified)              => orientation
      case (_, _, SpecifiedDirection.Flip)                     => orientation.invert
      case (_, _, SpecifiedDirection.Output)                   => orientation
      case (_, _, SpecifiedDirection.Input)                    => orientation.invert
      case other                                               => throw new Exception(s"Unexpected internal error! $other")
    }
  }


  def checkAnalog(as: Analog*)(implicit sourceInfo: SourceInfo): Unit = {
    val currentModule = Builder.currentModule.get.asInstanceOf[RawModule]
    try {
      as.toList match {
        case List(a) => BiConnect.markAnalogConnected(sourceInfo, a, DontCare, currentModule)
        case List(a, b) =>
          BiConnect.markAnalogConnected(sourceInfo, a, b, currentModule)
          BiConnect.markAnalogConnected(sourceInfo, b, a, currentModule)
      }
    } catch { // convert Exceptions to Builder.error's so compilation can continue
      case experimental.attach.AttachException(message) => Builder.error(message)
      case BiConnectException(message) => Builder.error(message)
    }
  }

  def assignAnalog(a: Analog, b: Data)(implicit sourceInfo: SourceInfo): Unit = {
    b match {
      case (ba: Analog) => {
        checkAnalog(a, ba)
        val currentModule = Builder.currentModule.get.asInstanceOf[RawModule]
        experimental.attach.impl(Seq(a, ba), currentModule)(sourceInfo)
      }
      case (DontCare) => {
        checkAnalog(a)
        pushCommand(DefInvalid(sourceInfo, a.lref))
      }
    }
  }
}
