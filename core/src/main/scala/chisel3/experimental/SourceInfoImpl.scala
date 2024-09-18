// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

private[chisel3] trait SourceLineImpl {
  def filename: String
  def line:     Int
  def col:      Int

  protected def _makeMessageImpl(f: String => String = x => x): String = f(s"@[${this.prettyPrint}]")

  def filenameOption: Option[String] = Some(filename)

  private def prettyPrint: String = {
    if (col == 0) s"$filename:$line" else s"$filename:$line:$col"
  }

  /** Convert to String for FIRRTL emission */
  def serialize: String = {
    if (col == 0) s"$filename $line" else s"$filename $line:$col"
  }
}

private[chisel3] trait ObjectSourceInfoImpl {

  /** Returns the best guess at the first stack frame that belongs to user code.
    */
  private def getUserLineNumber: Option[StackTraceElement] = {
    def isChiselClassname(className: String): Boolean = {
      // List of classpath prefixes that are Chisel internals and should be ignored when looking for user code
      // utils are not part of internals and errors there can be reported
      val chiselPrefixes = Set(
        "java.",
        "scala.",
        "chisel3.",
        "chisel3.internal.",
        "chisel3.experimental.",
        "chisel3.package$" // for some compatibility / deprecated types
      )
      !chiselPrefixes.filter(className.startsWith(_)).isEmpty
    }

    Thread
      .currentThread()
      .getStackTrace
      .toList
      .dropWhile(
        // Get rid of everything in Chisel core
        ste => isChiselClassname(ste.getClassName)
      )
      .headOption
  }

  private[chisel3] def materializeFromStacktrace: SourceInfo =
    getUserLineNumber match {
      case Some(elt) => new SourceLine(elt.getFileName, elt.getLineNumber, 0)
      case None      => UnlocatableSourceInfo
    }
}
