// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait Peekable$Intf[T <: Data] { self: Peekable[T] =>

  /** Get the value of a data port as a literal.
    * @return the value of the peeked data
    */
  def peek()(using sourceInfo: SourceInfo): T = _peekImpl()

  /** Expect the value of a data port to be equal to the expected value.
    * @param expected the expected value
    * @throws FailedExpectationException if the observed value does not match the expected value
    */
  def expect(expected: T)(using sourceInfo: SourceInfo): Unit = _expectImpl(expected)

  /** Expect the value of a data port to be equal to the expected value.
    * @param expected the expected value
    * @param message a message for the failure case
    * @throws FailedExpectationException if the observed value does not match the expected value
    */
  def expect(expected: T, message: String)(using sourceInfo: SourceInfo): Unit = _expectImpl(expected, message)
}

private[chisel3] trait TestableAggregate$Intf[T <: Aggregate] { self: TestableAggregate[T] =>

  /** Expect the value of a data port to be equal to the expected value, skipping all uninitialized elements.
    * @param expected the expected value
    * @param message a message for the failure case
    * @throws FailedExpectationException if the observed value does not match the expected value
    */
  def expectPartial(expected: T, message: String)(using sourceInfo: SourceInfo): Unit =
    _expectPartialImpl(expected, message)

  /** Expect the value of a data port to be equal to the expected value, skipping all uninitialized elements.
    * @param expected the expected value
    * @throws FailedExpectationException if the observed value does not match the expected value
    */
  def expectPartial(expected: T)(using sourceInfo: SourceInfo): Unit = _expectPartialImpl(expected)
}

private[chisel3] trait TestableElement$Intf[T <: Element] { self: TestableElement[T] =>

  /** Expect the value of a data port to be equal to the expected value.
    * @param expected the expected value
    * @param buildMessage a function taking (observedValue: T, expectedValue: T) and returning a String message for the failure case
    * @throws FailedExpectationException if the observed value does not match the expected value
    */
  def expect(expected: T, buildMessage: (T, T) => String)(using sourceInfo: SourceInfo): Unit =
    _expectWithBuildMessageImpl(expected, buildMessage)

  def expect(expected: BigInt)(using sourceInfo: SourceInfo): Unit = _expectBigIntImpl(expected)

  def expect(expected: BigInt, message: String)(using sourceInfo: SourceInfo): Unit =
    _expectBigIntImpl(expected, message)
}

private[chisel3] trait TestableBool$Intf { self: PeekPokeAPI.TestableBool =>

  def expect(value: Boolean)(using sourceInfo: SourceInfo): Unit = _expectBooleanImpl(value)
}
