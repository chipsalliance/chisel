// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3.experimental.SourceInfo

trait SimFailureException extends Exception with Serializable {
  def messages:   Seq[String]
  def sourceInfo: SourceInfo

  override def getMessage: String =
    (messages :+ sourceInfo.makeMessage(identity)).mkString(" ")
}

case class SVAssertionFailure(message: String)(implicit val sourceInfo: SourceInfo) extends SimFailureException {
  val messages = Seq(message)
}

case class TimedOutWaiting[T <: Serializable](
  cycles:       Int,
  condition:    T,
  extraMessage: Option[String] = None
)(
  implicit val sourceInfo: SourceInfo)
    extends SimFailureException {

  val messages = Seq(s"Timed out after $cycles cycles waiting for $condition.") ++ extraMessage
}
