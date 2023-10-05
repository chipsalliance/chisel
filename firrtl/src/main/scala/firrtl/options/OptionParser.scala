// SPDX-License-Identifier: Apache-2.0

package firrtl.options

import firrtl.AnnotationSeq

import scopt.OptionParser

case object OptionsHelpException extends Exception("Usage help invoked")

/** OptionParser mixin that causes the OptionParser to not call exit (call `sys.exit`) if the `--help` option is
  * passed
  */
trait DoNotTerminateOnExit { this: OptionParser[_] =>
  override def terminate(exitState: Either[String, Unit]): Unit = ()
}

/** OptionParser mixin that converts to [[OptionsException]]
  *
  * Scopt, by default, will print errors to stderr, e.g., invalid arguments will do this. However, a [[Stage]] uses
  * [[StageUtils.dramaticError]]. By converting this to an [[OptionsException]], a [[Stage]] can then catch the error an
  * convert it to an [[OptionsException]] that a [[Stage]] can get at.
  */
trait ExceptOnError { this: OptionParser[_] =>
  override def reportError(msg: String): Unit = throw new OptionsException(msg)
}

/** A modified OptionParser with mutable termination and additional checks
  */
trait DuplicateHandling extends OptionParser[AnnotationSeq] {

  override def parse(args: scala.collection.Seq[String], init: AnnotationSeq): Option[AnnotationSeq] = {

    /** Message for found duplicate options */
    def msg(x: String, y: String) = s"""Duplicate $x "$y" (did your custom Transform or OptionsManager add this?)"""

    val longDups = options.map(_.name).groupBy(identity).collect { case (k, v) if v.size > 1 && k != "" => k }
    val shortDups = options.map(_.shortOpt).flatten.groupBy(identity).collect { case (k, v) if v.size > 1 => k }

    if (longDups.nonEmpty) {
      throw new OptionsException(msg("long option", longDups.map("--" + _).mkString(",")), new IllegalArgumentException)
    }

    if (shortDups.nonEmpty) {
      throw new OptionsException(
        msg("short option", shortDups.map("-" + _).mkString(",")),
        new IllegalArgumentException
      )
    }

    super.parse(args, init)
  }

}
