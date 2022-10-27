// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.{prefix, BiConnect, Builder}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.{Analog, DataMirror, WaivedData}

import scala.collection.mutable
import chisel3.internal.ChildBinding
import firrtl.ir.Orientation
import chisel3.ActualDirection.Bidirectional

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
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessEq, Set.empty[Data], Set.empty[Data])
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
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonGreaterEq, Set.empty[Data], Set.empty[Data])
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
          DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessGreaterEq, Set.empty[Data], Set.empty[Data])
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
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonHashEq, Set.empty[Data], Set.empty[Data])
      }
    }
    final def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonHashEq, Set.empty[Data], Set.empty[Data])
      }
    }
    // Waivables
    final def :<=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonLessEq, Set.empty[Data], pWaived.waivers)
      }
    }
    final def :>=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonGreaterEq, Set.empty[Data], pWaived.waivers)
      }
    }
    final def :<>=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonLessGreaterEq, Set.empty[Data], pWaived.waivers)
      }
    }
    final def :#=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonHashEq, Set.empty[Data], pWaived.waivers)
      }
    }

    // Original non-erroring partial connect.. Do we dare?!?
    final def :<!>=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        val (cWaivers, pWaivers) = WaivedData.waiveUnmatched(consumer, producer)
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessGreaterEq, cWaivers.waivers, pWaivers.waivers)
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
    def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      for (a <- consumer) { a :#= DontCare }
    }

  }
  implicit class ConnectableDontCare(consumer: DontCare.type) {
    final def :>=[T <: Data](producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonGreaterEq, Set.empty[Data], Set.empty[Data])
      }
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

  def leafConnect(c: Data, p: Data, o: RelativeOrientation, op: ConnectionOperator)(implicit sourceInfo: SourceInfo): Unit = {
    (c, p, o, op.assignToConsumer, op.assignToProducer, op.alwaysAssignToConsumer) match {
      case (x: Analog, y: Analog, _, _, _, _) => assignAnalog(x, y)
      case (x: Analog, DontCare, _, _, _, _) => assignAnalog(x, DontCare)
      case (x, y, AlignedWithRoot(_), true, _, _) => c := p
      case (x, y, FlippedWithRoot(_), _, true, _) => p := c
      case (x, y, _, _, _, true) => c := p
      case other =>
    }
  }
  def startCoercing(d: Data): Boolean = {
    def recUp(x: Data): Boolean = x.binding match {
      case _ if isCoercing(x)    => true
      case Some(t: internal.TopBinding)   => false
      case Some(ChildBinding(p)) => recUp(p)
      case other => throw new Exception(s"Unexpected $other! $x, $d")
    }
    def isCoercing(d: Data): Boolean = {
      val s = DataMirror.specifiedDirectionOf(d)
      (s == SpecifiedDirection.Input) || (s == SpecifiedDirection.Output)
    }
    val ret = recUp(d)
    //println(s"startCoercing: $d gives $ret")
    ret
  }

  def doAssignment[T <: Data](consumer: T, producer: T, op: ConnectionOperator, cWaivers: Set[Data], pWaivers: Set[Data])(implicit sourceInfo: SourceInfo): Unit = {
    val errors = mutable.ArrayBuffer[String]()
    def doAssignment(c: Option[Data], co: RelativeOrientation, p: Option[Data], po: RelativeOrientation)(implicit sourceInfo: SourceInfo): Unit = {
      ((c, co), (p, po)) match {
        case ((None, _),                               (None, _))                                                                                           => ()
        // Operator waiver cases
        case ((Some(c), AlignedWithRoot(_)), (None, EmptyOrientation)) if op.assignToConsumer && op.noUnassigned && cWaivers.contains(c)                       => ()
        case ((Some(c), FlippedWithRoot(_)), (None, EmptyOrientation)) if op.assignToProducer && op.noDangles && cWaivers.contains(c)                          => ()
        case ((None, EmptyOrientation), (Some(p), AlignedWithRoot(_))) if op.assignToConsumer && op.noDangles && pWaivers.contains(p)                          => ()
        case ((None, EmptyOrientation), (Some(p), FlippedWithRoot(_))) if op.assignToProducer && op.noUnassigned && pWaivers.contains(p)                       => ()

        case ((Some(c), AlignedWithRoot(_)),              (None, EmptyOrientation))                if op.assignToConsumer && op.noUnassigned                   => errors += (s"unassigned consumer field $c")
        case ((Some(c: Aggregate), AlignedWithRoot(_)),   (None, EmptyOrientation))                                                                            => c.getElements.foreach(e => doAssignment(Some(e), deriveOrientation(e, consumer, co), None, EmptyOrientation))
        case ((Some(c), AlignedWithRoot(_)),              (None, EmptyOrientation))                if !op.mustMatch                                            => () // Dangling/unassigned consumer field, but we don't care

        case ((Some(c), FlippedWithRoot(_)),              (None, EmptyOrientation))                if op.assignToProducer && op.noDangles                      => errors += (s"dangling consumer field $c")
        case ((Some(c: Aggregate), FlippedWithRoot(_)),   (None, EmptyOrientation))                                                                            => c.getElements.foreach(e => doAssignment(Some(e), deriveOrientation(e, consumer, co), None, EmptyOrientation))
        case ((Some(c), FlippedWithRoot(_)),              (None, EmptyOrientation))                if !op.mustMatch                                            => () // Dangling/unassigned consumer field, but we don't care

        case ((None, EmptyOrientation),                (Some(p), AlignedWithRoot(_)))              if op.assignToConsumer && op.noDangles                      => errors += (s"dangling producer field $p")
        case ((None, EmptyOrientation),                (Some(p: Aggregate), AlignedWithRoot(_)))                                                               => p.getElements.foreach(e => doAssignment(None, EmptyOrientation, Some(e), deriveOrientation(e, producer, po)))
        case ((None, EmptyOrientation),                (Some(p), AlignedWithRoot(_)))              if !op.mustMatch                                            => () // Dangling/unassigned producer field, but we don't care

        case ((None, EmptyOrientation),                (Some(p), FlippedWithRoot(_)))              if op.assignToProducer && op.noUnassigned                   => errors += (s"unassigned producer field $p")
        case ((None, EmptyOrientation),                (Some(p: Aggregate), FlippedWithRoot(_)))                                                               => p.getElements.foreach(e => doAssignment(None, EmptyOrientation, Some(e), deriveOrientation(e, producer, po)))
        case ((None, EmptyOrientation),                (Some(p), FlippedWithRoot(_)))              if !op.mustMatch                                            => () // Dangling/unassigned producer field, but we don't care

        case ((Some(c), co),                           (None, po))                              if op.mustMatch                                             => errors += (s"unmatched consumer field $c")
        case ((None, co),                              (Some(p), po))                           if op.mustMatch                                             => errors += (s"unmatched producer field $p")

        case ((Some(c), co), (Some(p), po)) if (!co.alignsWith(po)) && (op.noWrongOrientations)                                                                       => errors += (s"inversely oriented fields $c and $p")
        case ((Some(c), co), (Some(p), po)) => (c, p) match {
          case (c: Record, p: Record) =>
            val cElements = c.elements
            val unusedPKeys = new mutable.LinkedHashSet[String]()
            val pElements = p.elements
            pElements.foreach {
              case (k, _) => unusedPKeys += k
            }
            val pKeys = pElements.keySet
            cElements.foreach { case (key, f) =>
              val pFOpt = pElements.get(key)
              val pFo = pFOpt.map { x =>
                unusedPKeys -= key
                deriveOrientation(x, producer, po)
              }.getOrElse(EmptyOrientation)
              doAssignment(Some(f), deriveOrientation(f, consumer, co), pFOpt, pFo)
            }
            unusedPKeys.foreach { pk =>
              val f = pElements(pk)
              doAssignment(None, EmptyOrientation, Some(f), deriveOrientation(f, producer, po))
            }
          case (c: Vec[Data @unchecked], p: Vec[Data @unchecked]) =>
            c.zip(p).foreach { case (cs, ps) =>
              // Because Chisel is awful, you can do Vec(Flipped(UInt))
              doAssignment(Some(cs), deriveOrientation(cs, consumer, co), Some(ps), deriveOrientation(ps, producer, po))
            }
            if(c.size > p.size) {
              c.getElements.slice(p.size, c.size).foreach { cs =>
                doAssignment(Some(cs), deriveOrientation(cs, consumer, co), None, EmptyOrientation)
              }
            }
            if(c.size < p.size) {
              p.getElements.slice(c.size, p.size).foreach { ps =>
                doAssignment(None, EmptyOrientation, Some(ps), deriveOrientation(ps, producer, po))
              }
            }
          // Am matching orientation of the non-DontCare, regardless
          case (c: Aggregate, DontCare) => c.getElements.foreach { case f => doAssignment(Some(f), deriveOrientation(f, consumer, co), Some(DontCare), deriveOrientation(f, consumer, co)) }
          case (DontCare, p: Aggregate) => p.getElements.foreach { case f => doAssignment(Some(DontCare), deriveOrientation(f, producer, po), Some(f), deriveOrientation(f, producer, po)) }
          case (c, p) if  co.alignsWith(po) => leafConnect(c, p, co, op)
          case (c, p) if !co.alignsWith(po) && op.assignToConsumer && !op.assignToProducer => leafConnect(c, p, co, op)
          case (c, p) if !co.alignsWith(po) && !op.assignToConsumer && op.assignToProducer => leafConnect(c, p, po, op)
        }
        case other => throw new Exception(other.toString + " " + op)
      }
    }
    doAssignment(Some(consumer), AlignedWithRoot(startCoercing(consumer)), Some(producer), AlignedWithRoot(startCoercing(producer)))
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
  def assign[T <: Data](cRoot: T, pRoot: T, cOp: ConnectionOperator, cWaivers: Set[Data], pWaivers: Set[Data])(implicit sourceInfo: SourceInfo): Unit = {
    doAssignment(cRoot, pRoot, cOp, cWaivers, pWaivers)
  }


  // Indicates whether the active side is aligned or flipped relative to the active side's root
  sealed trait RelativeOrientation { 
    def invert: RelativeOrientation
    def isCoercing: Boolean
    def coerce: RelativeOrientation
    def alignsWith(o: RelativeOrientation): Boolean = o.coerce == this.coerce // Clear out coerce in comparison
  }
  case class AlignedWithRoot(isCoercing: Boolean) extends RelativeOrientation {
    def invert = if(isCoercing) this else FlippedWithRoot(isCoercing)
    def coerce = this.copy(true)
  }
  case class FlippedWithRoot(isCoercing: Boolean) extends RelativeOrientation {
    def invert = if(isCoercing) this else AlignedWithRoot(isCoercing)
    def coerce = this.copy(true)
  }
  case object EmptyOrientation extends RelativeOrientation {
    def invert = this
    def isCoercing = false
    def coerce = this
  }


  sealed trait ConnectionOperator {
    val noDangles: Boolean
    val noUnassigned: Boolean
    val mustMatch: Boolean
    val noWrongOrientations: Boolean
    val assignToConsumer: Boolean
    val assignToProducer: Boolean
    val alwaysAssignToConsumer: Boolean
  }
  case object ColonLessEq extends ConnectionOperator {
    val noDangles: Boolean = false
    val noUnassigned: Boolean = true
    val mustMatch: Boolean = false
    val noWrongOrientations: Boolean = true
    val assignToConsumer: Boolean = true
    val assignToProducer: Boolean = false
    val alwaysAssignToConsumer: Boolean = false
  }
  case object ColonGreaterEq extends ConnectionOperator {
    val noDangles: Boolean = false
    val noUnassigned: Boolean = true
    val mustMatch: Boolean = false
    val noWrongOrientations: Boolean = true
    val assignToConsumer: Boolean = false
    val assignToProducer: Boolean = true
    val alwaysAssignToConsumer: Boolean = false
  }
  case object ColonLessGreaterEq extends ConnectionOperator {
    val noDangles: Boolean = true
    val noUnassigned: Boolean = true
    val mustMatch: Boolean = true
    val noWrongOrientations: Boolean = true
    val assignToConsumer: Boolean = true
    val assignToProducer: Boolean = true
    val alwaysAssignToConsumer: Boolean = false
  }
  case object ColonHashEq extends ConnectionOperator {
    val noDangles: Boolean = true
    val noUnassigned: Boolean = true
    val mustMatch: Boolean = true
    val noWrongOrientations: Boolean = false
    val assignToConsumer: Boolean = true
    val assignToProducer: Boolean = false
    val alwaysAssignToConsumer: Boolean = true
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
    //TODO(azidar): write exhaustive tests to demonstrate Chisel and chisel3 type/direction declarations compose
    val x = (DataMirror.specifiedDirectionOf(subMember)) match {
      case (SpecifiedDirection.Unspecified) => orientation
      case (SpecifiedDirection.Flip)        => orientation.invert
      case (SpecifiedDirection.Output)      => orientation.coerce
      case (SpecifiedDirection.Input)       => orientation.invert.coerce
      case other                            => throw new Exception(s"Unexpected internal error! $other")
    }
    //println(s"$subMember has $x")
    x
  }

  // Adam's Commandments
  /*

  The 'states' of a chiselType

  1. A pure type
  2. A child field of a Bundle type
  3. A bound thing

  Bool()                         // 0. "Pure type"
  Aligned(Bool())                // 1. "Incomplete thing"
  Aligned(Aligned(Bool()))       // 2. "Incomplete thing", same as above, Aligned can take 0 or 1
  (new Decoupled().b)            // 3. "direction of me relative to my parent is now known"
  Aligned((new Decoupled().b))   // 4. ERROR!

  Wire(Aligned(new Decoupled())) // 5. ERROR!
  Wire(Flipped(new Decoupled())) // 6. ERROR!

class Decoupled(gen: () => T) {
  val x = gen()
  val ready = Flipped(Bool())
  val valid = Aligned(Bool())

}


y 1. Only child bindings can have specified direction
y 2. "incoming ports" are Incoming(chiselType) with IncomingPortBinding, "outgoing ports" are Outgoing(chiselType) with OutgoingPortBinding
    'chiselType' does not have a specified direction
y 3. Add a new function "stampSamePort" to create a new Incoming/Outgoing identical port because that info is not in the type, (it is in the binding)
y 4. Add Aligned(x), Flipped(x) which are called when creating a field of a bundle/record
    - sets isFlipped
  5. Data has two set-once vars:
    - isFlipped, and can have one of 2 values (indicated by the above), is known when it is child of a bundle
    - coerceNoFlips, can can either be true or false, (indicates that the type actually ignores children isFlipped value when determining "resolve direction")
    - we keeps these separate so our clonetype function is easier to implement (and we don't lose information)
  6. "cloning a type"
    - on Bundle will copy iAmAFieldAndIAmFlipped and childrenAreCoerced direction of each original element to each cloned element
    - on Vec will copy isCoerced from original sample_element to cloned sample_element, with not copy isFlipped
    - on Element will not copy isCoerced nor isFlipped
  7. "binding a type"
  7. "resolve direction" gives a "aligned/flipped" value for members of a Data that is bound (known purely from parent to child)
    - 
  8. "port direction" is only on Data bound as a port, and combines the "resolve direction" with binding (incoming/outgoing) to give incoming/outgoing of submember
  9. stripsFlipsOf(x) creates a new ChiselType (like chiselTypeOf)
    - is the same scala type
    - "clones the type" and then recursively sets all coerceNoFlips fields to be true

  class Bundle {
    val r = Aligned(coerce(Decoupled))
    val x = Flipped(coerce(Decoupled)) isFlipped = 1, isCoerced = 1
  }
  val o = originalChiselTypeOf(b.x)
  
  val r = (Decoupled)
  val r = Reg(coerce(Decoupled))
  r.
  */

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
  def isInvalid(d: Data)(implicit sourceInfo: SourceInfo): Unit = {
    pushCommand(DefInvalid(sourceInfo, d.lref))
  }
}
