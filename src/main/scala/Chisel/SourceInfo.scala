package Chisel

import scala.language.experimental.macros

// A box around the details
// Nervous about having an implicit Option[_] in case of implicit None
case class SourceInfo(sinfo: Option[SourceInfo.SourceLoc])
object SourceInfo {
  sealed trait SourceLoc
  case class FromFile(file: String, line: String) extends SourceLoc

  implicit def materialize: SourceInfo = macro Locators.injectSourceInfo

  // Import this to suppress enclosure info details
  object Suppress {
    implicit val Implicitly = SourceInfo(None)
  }
}

// Not for use by users directly!
object Locators {
  import scala.reflect.macros.blackbox.Context

  def injectSourceInfo(c: Context): c.Tree = {
    import c.universe._
    val file = c.enclosingPosition.source.file.name.toString
    val line = c.enclosingPosition.line.toString
    val info = q"_root_.Chisel.SourceInfo.FromFile($file,$line)"
    // sanitized SourceInfo(Some(SourceInfo.FromFile(file, line)))
    q"_root_.Chisel.SourceInfo(_root_.scala.Some($info))"
  }
}
