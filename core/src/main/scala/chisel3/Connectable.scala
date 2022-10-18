// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.{prefix, BiConnect, Builder}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.{Analog, DataMirror, Waivable}
import chisel3.experimental.Defaulting._

import scala.collection.mutable
import chisel3.internal.ChildBinding
import firrtl.ir.Orientation

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
    final def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
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
    def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      for (a <- consumer) { a :#= DontCare }
    }
  }
  implicit class ConnectableDontCare(consumer: DontCare.type) {
    final def :>=[T <: Data](producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonGreaterEq)
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
  def leafConnect(c: Data, p: Data, o: RelativeOrientation, op: ConnectionOperator)(implicit sourceInfo: SourceInfo): Unit = {
    (c, p, o, op.assignToConsumer, op.assignToProducer, op.alwaysAssignToConsumer) match {
      case (x: Aggregate, y, _, _, _, _) => Builder.error(s"Internal error! Unexpected Aggregate $x!")
      case (x, y: Aggregate, _, _, _, _) => Builder.error(s"Internal error! Unexpected Aggregate $y!")
      case (x: Analog, y: Analog, _, _, _, _) => assignAnalog(x, y)
      case (x: Analog, DontCare, _, _, _, _) => assignAnalog(x, DontCare)
      case (x, y, AlignedWithRoot, true, _, _) => c := p
      case (x, y, FlippedWithRoot, _, true, _) => p := c
      case (x, y, _, _, _, true) => c := p
      case other =>
    }
  }


  def doAssignment(consumer: Data, producer: Data, op: ConnectionOperator)(implicit sourceInfo: SourceInfo): Unit = {
    val errors = mutable.ArrayBuffer[String]()
    def doAssignment(c: Option[Data], co: RelativeOrientation, p: Option[Data], po: RelativeOrientation)(implicit sourceInfo: SourceInfo): Unit = {
      ((c, co), (p, po)) match {
        case ((None, _),                               (None, _))                                                                                           => ()
        case ((Some(c: Waivable[_]), AlignedWithRoot), (None, EmptyOrientation))                if op.assignToConsumer && op.noUnassigned && c.okToUnassign => ()
        case ((Some(c: Waivable[_]), FlippedWithRoot), (None, EmptyOrientation))                if op.assignToProducer && op.noDangles && c.okToDangle      => ()
        case ((None, EmptyOrientation),                (Some(p: Waivable[_]), AlignedWithRoot)) if op.assignToConsumer && op.noDangles && p.okToDangle      => ()
        case ((None, EmptyOrientation),                (Some(p: Waivable[_]), FlippedWithRoot)) if op.assignToProducer && op.noUnassigned && p.okToUnassign => ()

        case ((Some(c), AlignedWithRoot),              (None, EmptyOrientation))                if op.assignToConsumer && op.noUnassigned                   => errors += (s"unassigned consumer field $c")
        case ((Some(c: Aggregate), AlignedWithRoot),   (None, EmptyOrientation))                                                                            => c.getElements.foreach(e => doAssignment(Some(e), deriveOrientation(e, consumer, co), None, EmptyOrientation))
        case ((Some(c), AlignedWithRoot),              (None, EmptyOrientation))                if !op.mustMatch                                            => () // Dangling/unassigned consumer field, but we don't care

        case ((Some(c), FlippedWithRoot),              (None, EmptyOrientation))                if op.assignToProducer && op.noDangles                      => errors += (s"dangling consumer field $c")
        case ((Some(c: Aggregate), FlippedWithRoot),   (None, EmptyOrientation))                                                                            => c.getElements.foreach(e => doAssignment(Some(e), deriveOrientation(e, consumer, co), None, EmptyOrientation))
        case ((Some(c), FlippedWithRoot),              (None, EmptyOrientation))                if !op.mustMatch                                            => () // Dangling/unassigned consumer field, but we don't care

        case ((None, EmptyOrientation),                (Some(p), AlignedWithRoot))              if op.assignToConsumer && op.noDangles                      => errors += (s"dangling producer field $p")
        case ((None, EmptyOrientation),                (Some(p: Aggregate), AlignedWithRoot))                                                               => p.getElements.foreach(e => doAssignment(None, EmptyOrientation, Some(e), deriveOrientation(e, producer, po)))
        case ((None, EmptyOrientation),                (Some(p), AlignedWithRoot))              if !op.mustMatch                                            => () // Dangling/unassigned producer field, but we don't care

        case ((None, EmptyOrientation),                (Some(p), FlippedWithRoot))              if op.assignToProducer && op.noUnassigned                   => errors += (s"unassigned producer field $p")
        case ((None, EmptyOrientation),                (Some(p: Aggregate), FlippedWithRoot))                                                               => p.getElements.foreach(e => doAssignment(None, EmptyOrientation, Some(e), deriveOrientation(e, producer, po)))
        case ((None, EmptyOrientation),                (Some(p), FlippedWithRoot))              if !op.mustMatch                                            => () // Dangling/unassigned producer field, but we don't care

        case ((Some(c), co),                           (None, po))                              if op.mustMatch                                             => errors += (s"unmatched consumer field $c")
        case ((None, co),                              (Some(p), po))                           if op.mustMatch                                             => errors += (s"unmatched producer field $p")

        case ((Some(c), co), (Some(p), po)) if (co != po) && (op.noWrongOrientations)                                                                       => errors += (s"inversely oriented fields $c and $p")
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
              doAssignment(Some(cs), co, Some(ps), po)
            }
            if(c.size > p.size) {
              c.getElements.slice(p.size, c.size).foreach { cs =>
                doAssignment(Some(cs), co, None, EmptyOrientation)
              }
            }
            if(c.size < p.size) {
              p.getElements.slice(c.size, p.size).foreach { ps =>
                doAssignment(None, EmptyOrientation, Some(ps), po)
              }
            }
          case (c: Aggregate, DontCare) => c.getElements.foreach { case f => doAssignment(Some(f), deriveOrientation(f, consumer, co), Some(DontCare), po) }
          case (DontCare, p: Aggregate) => p.getElements.foreach { case f => doAssignment(Some(DontCare), co, Some(f), deriveOrientation(f, producer, po)) }
          case (c, p) if co == po => leafConnect(c, p, co, op)
          case (c, p) if co != po && op.assignToConsumer && !op.assignToProducer => leafConnect(c, p, co, op)
          case (c, p) if co != po && !op.assignToConsumer && op.assignToProducer => leafConnect(c, p, po, op)
        }
        case other => throw new Exception(other.toString + " " + op)
      }
    }
    doAssignment(Some(consumer), AlignedWithRoot, Some(producer), AlignedWithRoot)
    if(errors.nonEmpty) {
      Builder.error(errors.mkString("\n"))
    }
  }

  def doAssignment(trie: Trie[String, LeafConnection], op: ConnectionOperator)(implicit sourceInfo: SourceInfo): Unit = {
    val errors = mutable.ArrayBuffer[String]()
    trie.collectDeep {
      case (path, Some(LeafConnection(Some((c, co)), Some((p, po))))) if co == po => leafConnect(c, p, co, op)
      case (path, Some(LeafConnection(Some((c, co)), Some((p, po))))) if co != po =>
        if(op.noWrongOrientations) errors += (s"inversely oriented fields $c and $p") else {
          if(op.assignToConsumer) leafConnect(c, p, co, op)
          if(op.assignToProducer) leafConnect(c, p, po, op)
        }
      case (path, Some(LeafConnection(Some((c, FlippedWithRoot)), None))) if(op.noDangles || op.mustMatch) => errors += (s"dangling consumer field $c")
      case (path, Some(LeafConnection(None, Some((p, AlignedWithRoot))))) if(op.noDangles || op.mustMatch) => errors += (s"dangling producer field $p")
      // Defaulting case
      case (path, Some(LeafConnection(Some((c, AlignedWithRoot)), None))) if c.hasConnectableDefault => leafConnect(c, c.connectableDefault, AlignedWithRoot, op)
      case (path, Some(LeafConnection(None, Some((p, FlippedWithRoot))))) if p.hasConnectableDefault => leafConnect(p.connectableDefault, p, FlippedWithRoot, op)

      // Non-defaulting case
      case (path, Some(LeafConnection(Some((c, AlignedWithRoot)), None))) if (op.noUnassigned && op.assignToConsumer) => errors += (s"unassigned consumer field $c")
      case (path, Some(LeafConnection(Some((c, AlignedWithRoot)), None))) if (op.mustMatch) => errors += (s"unmatched consumer field $c")
      case (path, Some(LeafConnection(None, Some((p, FlippedWithRoot))))) if (op.noUnassigned && op.assignToProducer) => errors += (s"unassigned producer field $p")
      case (path, Some(LeafConnection(None, Some((p, FlippedWithRoot))))) if (op.mustMatch) => errors += (s"unmatched producer field $p")
      case (path, Some(other)) => //throw new Exception("BAD!! Unreachable code is reached, something went wrong")
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
    //val trie = buildTrie(cRoot, pRoot)
    //doAssignment(trie, cOp)
    doAssignment(cRoot, pRoot, cOp)
  }


  // Indicates whether the active side is aligned or flipped relative to the active side's root
  sealed trait RelativeOrientation { def invert: RelativeOrientation }
  case object AlignedWithRoot extends RelativeOrientation { def invert = FlippedWithRoot }
  case object FlippedWithRoot extends RelativeOrientation { def invert = AlignedWithRoot }
  case object EmptyOrientation extends RelativeOrientation { def invert = EmptyOrientation }


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
    (root.direction, subMember.direction, DataMirror.specifiedDirectionOf(subMember)) match {
      //case (ActualDirection.Output, ActualDirection.Output, _) => AlignedWithRoot
      //case (ActualDirection.Input, ActualDirection.Input, _)   => AlignedWithRoot
      //case (ActualDirection.Output, ActualDirection.Input, _)  => FlippedWithRoot
      //case (ActualDirection.Input, ActualDirection.Output, _)  => FlippedWithRoot
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
  def isInvalid(d: Data)(implicit sourceInfo: SourceInfo): Unit = {
    pushCommand(DefInvalid(sourceInfo, d.lref))
  }
}
