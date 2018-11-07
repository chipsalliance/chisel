// See LICENSE for license details.

package firrtl.options

import firrtl.{FIRRTLException, AnnotationSeq}

import scopt.OptionParser

/** Causes an OptionParser to not call exit (call `sys.exit`) if the `--help` option is passed
  */
trait DoNotTerminateOnExit { this: OptionParser[_] =>
  override def terminate(exitState: Either[String, Unit]): Unit = Unit
}

/** A modified OptionParser with mutable termination and additional checks
  */
trait DuplicateHandling extends OptionParser[AnnotationSeq] {

  override def parse(args: Seq[String], init: AnnotationSeq): Option[AnnotationSeq] = {

    /** Message for found duplicate options */
    def msg(x: String, y: String) = s"""Duplicate $x "$y" (did your custom Transform or OptionsManager add this?)"""

    val longDups = options.map(_.name).groupBy(identity).collect{ case (k, v) if v.size > 1 && k != "" => k }
    val shortDups = options.map(_.shortOpt).flatten.groupBy(identity).collect{ case (k, v) if v.size > 1 => k }


    if (longDups.nonEmpty)  {
      throw new OptionsException(msg("long option", longDups.map("--" + _).mkString(",")), new IllegalArgumentException)
    }

    if (shortDups.nonEmpty) {
      throw new OptionsException(msg("short option", shortDups.map("-" + _).mkString(",")), new IllegalArgumentException)
    }

    super.parse(args, init)
  }

}
