// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.{BiConnect, Builder}
import chisel3.experimental.{prefix, SourceInfo}

package object connectable {

  import Connection.connect

  type ConnectableDocs = Connectable.ConnectableDocs

  /** Connectable Typeclass defines the following operators on all subclasses of Data: :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableOperators[T <: Data](consumer: T)
      extends Connectable.ConnectableOpExtension(Data.makeConnectableDefault(consumer))

  /** ConnectableVec Typeclass defines the following operators on between a (consumer: Vec) and (producer: Seq): :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableVecOperators[T <: Data](consumer: Vec[T]) extends ConnectableDocs {

    /** Shared implementation for the Vec/Seq connection operators.
      *
      * @param producer the right-hand-side of the connection
      * @param op the connection operator to apply to each pair of elements
      */
    private def connectSeq(producer: Seq[T])(op: (T, T) => Unit)(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { op(a, b) }
    }

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    def :<=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = connectSeq(producer)(_ :<= _)

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    def :>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = connectSeq(producer)(_ :>= _)

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    def :<>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = connectSeq(producer)(_ :<>= _)

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    def :#=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = connectSeq(producer)(_ :#= _)

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      for (a <- consumer) { a :#= DontCare }
    }
  }

  /** ConnectableOption Typeclass defines the following operators on between a (consumer: Option[T]) and (producer: Option[T]): :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableOptionOperators[T <: Data](consumer: Option[T]) extends ConnectableDocs {

    /** Shared implementation for the Option connection operators.
      *
      * @param producer the right-hand-side of the connection
      * @param op the connection operator to apply when both consumer and producer are non-empty
      */
    private def connectOption(
      producer: Option[T]
    )(op: (T, T) => Unit)(implicit sourceInfo: SourceInfo): Unit = (consumer, producer) match {
      case (Some(c), Some(p)) => op(c, p)
      case (None, None)       => ()
      case _ =>
        Builder.error(
          s"Connecting Options of different emptiness is not allowed: consumer is $consumer, producer is $producer"
        )
    }

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    def :<=(producer: Option[T])(implicit sourceInfo: SourceInfo): Unit = connectOption(producer)(_ :<= _)

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    def :>=(producer: Option[T])(implicit sourceInfo: SourceInfo): Unit = connectOption(producer)(_ :>= _)

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    def :<>=(producer: Option[T])(implicit sourceInfo: SourceInfo): Unit = connectOption(producer)(_ :<>= _)

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    def :#=(producer: Option[T])(implicit sourceInfo: SourceInfo): Unit = connectOption(producer)(_ :#= _)

    /** $colonHashEq
      *
      * If the consumer is empty, this is a no-op.
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = consumer match {
      case Some(c) => c :#= DontCare
      case None    => ()
    }
  }

  implicit class ConnectableDontCare(consumer: DontCare.type) extends ConnectableDocs {

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[T <: Data](producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        connect(consumer, producer, ColonGreaterEq)
      }
    }
  }

  /** ConnectableBits Typeclass defines the :%= operator on UInt and SInt: an explicit truncating connection operator
  *
  * @param consumer the left-hand-side of the connection
  */
  implicit class ConnectableBitsOperators[T <: Bits](consumer: T)
      extends Connectable.ConnectableBitsOpExtension(Data.makeConnectableDefault(consumer))

}
