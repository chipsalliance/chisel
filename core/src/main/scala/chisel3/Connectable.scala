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

object Connectable {

  /** The default connection operators for Chisel hardware components
   * 
   * @define colonHashEq The "mono-direction connection operator", aka the "coercion operator".
   * 
   * For `consumer :#= producer`, all leaf fields of consumer (regardless of relative flip) are driven by the corresponding leaf fields of producer (regardless of relative flip)
   * 
   * Identical to calling :<= and :>=, but swapping consumer/producer for :>= (order is irrelevant), e.g.:
   *   consumer :<= producer
   *   producer :>= consumer
   * 
   * $chiselTypeRestrictions
   * 
   * Additional notes:
   * - Connecting two [[util.DecoupledIO]]'s would connect `bits`, `valid`, AND `ready` from producer to consumer (despite `ready` being flipped)
   * - Functionally equivalent to chisel3.:=, but different than Chisel.:=
   *
   * @group connection
   * 
   * @define colonLessEq The "aligned connection operator" between a producer and consumer.
   *
   * For `consumer :<= producer`, each of `consumer`'s leaf fields which are aligned with respect to `consumer` are driven from the corresponding `producer` leaf field.
   * Only `consumer`'s leaf/branch alignments influence the connection.
   * 
   * $chiselTypeRestrictions
   * 
   * Additional notes:
   *  - Connecting two [[util.DecoupledIO]]'s would connect `bits` and `valid` from producer to consumer, but leave `ready` unconnected
   *
   * @group connection
   * 
   * @define colonGreaterEq The "flipped connection operator", or the "backpressure connection operator" between a producer and consumer.
   *
   * For `consumer :>= producer`, each of `producer`'s leaf fields which are flipped with respect to `producer` are driven from the corresponding consumer leaf field
   * Only `producer`'s leaf/branch alignments influence the connection.
   * 
   * $chiselTypeRestrictions
   *
   * Additional notes:
   *  - Connecting two [[util.DecoupledIO]]'s would connect `ready` from consumer to producer, but leave `bits` and `valid` unconnected
   *
   * @group connection
   *
   * @define colonLessGreaterEq The "bi-direction connection operator", aka the "tur-duck-en operator"
   *
   * For `consumer :<>= producer`, both producer and consumer leafs could be driving or be driven-to.
   * The `consumer`'s fields aligned w.r.t. `consumer` will be driven by corresponding fields of `producer`;
   * the `producer`'s fields flipped w.r.t. `producer` will be driven by corresponding fields of `consumer`
   *
   * Identical to calling `:<=` and `:>=` in sequence (order is irrelevant), e.g. `consumer :<= producer` then `consumer :>= producer`
   * 
   * $chiselTypeRestrictions
   * - An additional type restriction is that all relative orientations of `consumer` and `producer` must match exactly.
   * 
   * Additional notes:
   *  - Connecting two wires of [[util.DecoupledIO]] chisel type would connect `bits` and `valid` from producer to consumer, and `ready` from consumer to producer.
   *  - If the types of consumer and producer also have identical relative flips, then we can emit FIRRTL.<= as it is a stricter version of chisel3.:<>=
   *  - "turk-duck-en" is a dish where a turkey is stuffed with a duck, which is stuffed with a chicken; `:<>=` is a `:=` stuffed with a `<>`
   *
   * @define chiselTypeRestrictions The following restrictions apply:
   *  - The Chisel type of consumer and producer must be the "same shape" recursively:
   *    - All ground types are the same (UInt and UInt are same, SInt and UInt are not), but widths can be different (implicit trunction/padding occurs)
   *    - All vector types are the same length
   *    - All bundle types have the same field names, but the flips of fields can be different between producer and consumer
   *  - The leaf fields that are ultimately assigned to, must be assignable. This means they cannot be module inputs or instance outputs.
   */
  trait ConnectableDocs

  /** ConnectableData Typeclass defines the following operators on all subclasses of Data: :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableData[T <: Data](consumer: T) extends ConnectableDocs {

    /** $colonLessEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=[S <: Data](producer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessEq, Set.empty[Data], Set.empty[Data])
      }
    }

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[S <: Data](producer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonGreaterEq, Set.empty[Data], Set.empty[Data])
      }
    }

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=[S <: Data](producer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        val evaluatedProducer = producer
        val canFirrtlConnect =
          try {
            BiConnect.canFirrtlConnectData(
              consumer,
              evaluatedProducer,
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
          consumer.firrtlConnect(evaluatedProducer)
        } else {
          // cannot call :<= and :>= directly because otherwise prefix is called twice
          DirectionalConnectionFunctions.assign(consumer, evaluatedProducer, DirectionalConnectionFunctions.ColonLessGreaterEq, Set.empty[Data], Set.empty[Data])
        }
      }
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      */
    final def :#=[S <: Data](producer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonHashEq, Set.empty[Data], Set.empty[Data])
      }
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      */
    final def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonHashEq, Set.empty[Data], Set.empty[Data])
      }
    }

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=[S <: Data](pWaived: WaivedData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonLessEq, Set.empty[Data], pWaived.waivers)
      }
    }

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[S <: Data](pWaived: WaivedData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonGreaterEq, Set.empty[Data], pWaived.waivers)
      }
    }

    /** $colonLessGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=[S <: Data](pWaived: WaivedData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonLessGreaterEq, Set.empty[Data], pWaived.waivers)
      }
    }

    /** $colonHashEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      */
    final def :#=[S <: Data](pWaived: WaivedData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonHashEq, Set.empty[Data], pWaived.waivers)
      }
    }

    ///** $colonLessBangGreaterEq
    //  * 
    //  * @param producer the right-hand-side of the connection
    //  */
    //final def :<!>=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
    //  prefix(consumer) {
    //    //val (cWaivers, pWaivers) = WaivedData.waiveUnmatched(consumer, producer)
    //    val cWaivers = consumer.waiveAll
    //    val pWaivers = producer.waiveAll
    //    DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessGreaterEq, cWaivers.waivers, pWaivers.waivers)
    //  }
    //}
  }

  /** ConnectableVec Typeclass defines the following operators on between a (consumer: Vec) and (producer: Seq): :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableVec[T <: Data](consumer: Vec[T]) {

    /** $colonLessEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    def :<=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :<= b }
    }

    /** $colonGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    def :>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :>= b }
    }

    /** $colonLessGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    def :<>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :<>= b }
    }

    /** $colonHashEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      */
    def :#=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :#= b }
    }

    /** $colonHashEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      */
    def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      for (a <- consumer) { a :#= DontCare }
    }
  }

  implicit class ConnectableDontCare(consumer: DontCare.type) {
    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[T <: Data](producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonGreaterEq, Set.empty[Data], Set.empty[Data])
      }
    }
  }
}

private[chisel3] object DirectionalConnectionFunctions {
  // Consumed by the := operator, set to what chisel3 will eventually become.
  import RelativeOrientation.RelativeOrientationMatchingZipOfChildren.matchingZipOfChildren
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
      case (x, y, _: AlignedWithRoot, true, _, _) => c := p
      case (x, y, _: FlippedWithRoot, _, true, _) => p := c
      case (x, y, _, _, _, true) => c := p
      case other =>
    }
  }
  def doAssignment[T <: Data](consumer: T, producer: T, op: ConnectionOperator, cWaivers: Set[Data], pWaivers: Set[Data])(implicit sourceInfo: SourceInfo): Unit = {
    val errors = mutable.ArrayBuffer[String]()
    import RelativeOrientation.deriveOrientation
    def doAssignment(co: RelativeOrientation, po: RelativeOrientation)(implicit sourceInfo: SourceInfo): Unit = {
      (co, po) match {
        // Base Case 0: should probably never happen
        case (EmptyOrientation, EmptyOrientation) => ()

        // Base Case 1: early exit if dangling/unassigned is wavied
        case (co: NonEmptyOrientation, EmptyOrientation) if co.isWaived => ()
        case (EmptyOrientation, po: NonEmptyOrientation) if po.isWaived => ()

        // Base Case 2: early exit if operator requires matching orientations, but they don't align
        case (co: NonEmptyOrientation, po: NonEmptyOrientation) if (!co.alignsWith(po)) && (op.noWrongOrientations) => errors += (s"inversely oriented fields ${co.data} and ${po.data}")

        // Base Case 3: operator error on dangling/unassigned fields
        case (c: NonEmptyOrientation, EmptyOrientation) => errors += (s"${c.errorWord(op)} consumer field ${co.data}")
        case (EmptyOrientation, p: NonEmptyOrientation) => errors += (s"${p.errorWord(op)} producer field ${po.data}")

        // Recursive Case 4: non-empty orientations
        case (co: NonEmptyOrientation, po: NonEmptyOrientation) => 
          (co.data, po.data) match {
          case (c: Record, p: Record) =>
            matchingZipOfChildren(Some(co), Some(po)).foreach {
              case (ceo, peo) => doAssignment(ceo.getOrElse(EmptyOrientation), peo.getOrElse(EmptyOrientation))
            }
          case (c: Vec[Data @unchecked], p: Vec[Data @unchecked]) =>
            matchingZipOfChildren(Some(co), Some(po)).foreach {
              case (ceo, peo) => doAssignment(ceo.getOrElse(EmptyOrientation), peo.getOrElse(EmptyOrientation))
            }
          // Am matching orientation of the non-DontCare, regardless
          case (c: Aggregate, DontCare) => c.getElements.foreach { case f => doAssignment(deriveOrientation(f, co), deriveOrientation(f, co).swap(DontCare)) }
          case (DontCare, p: Aggregate) => p.getElements.foreach { case f => doAssignment(deriveOrientation(f, po).swap(DontCare), deriveOrientation(f, po)) }
          case (c, p) if  co.alignsWith(po) => leafConnect(c, p, co, op)
          case (c, p) if !co.alignsWith(po) && op.assignToConsumer && !op.assignToProducer => leafConnect(c, p, co, op)
          case (c, p) if !co.alignsWith(po) && !op.assignToConsumer && op.assignToProducer => leafConnect(c, p, po, op)
        }
        case other => throw new Exception(other.toString + " " + op)
      }
    }
    doAssignment(RelativeOrientation(consumer, cWaivers, true), RelativeOrientation(producer, pWaivers, false))
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
    val noDangles: Boolean = true
    val noUnassigned: Boolean = true
    val mustMatch: Boolean = true
    val noWrongOrientations: Boolean = true
    val assignToConsumer: Boolean = true
    val assignToProducer: Boolean = false
    val alwaysAssignToConsumer: Boolean = false
  }
  case object ColonGreaterEq extends ConnectionOperator {
    val noDangles: Boolean = true
    val noUnassigned: Boolean = true
    val mustMatch: Boolean = true
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
