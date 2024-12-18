// SPDX-License-Identifier: Apache-2.0

package firrtl.annotations

import java.io.File

import firrtl.FirrtlUserException

/** Representation of the two types of `readmem` statements available in Verilog.
  */
sealed abstract class MemoryLoadFileType(val value: String) {
  def serialize: String = value
}

object MemoryLoadFileType {
  // purely for backwards compatibility with chisel3's ChiselLoadMemoryAnnotation
  type FileType = MemoryLoadFileType

  case object Hex extends MemoryLoadFileType("h")
  case object Binary extends MemoryLoadFileType("b")
  def deserialize(s: String): MemoryLoadFileType = s match {
    case "h" => MemoryLoadFileType.Hex
    case "b" => MemoryLoadFileType.Binary
    case _   => throw new FirrtlUserException(s"Unrecognized MemoryLoadFileType: $s")
  }
}

/** Firrtl implementation for load memory
  * @param target        memory to load
  * @param fileName      name of input file
  * @param hexOrBinary   use `\$readmemh` or `\$readmemb`
  */
case class LoadMemoryAnnotation(
  target:                ComponentName,
  fileName:              String,
  hexOrBinary:           MemoryLoadFileType = MemoryLoadFileType.Hex,
  originalMemoryNameOpt: Option[String] = None)
    extends SingleTargetAnnotation[Named] {

  val (prefix, suffix) = {
    fileName.split("""\.""").toList match {
      case Nil =>
        throw new Exception(s"empty filename not allowed in LoadMemoryAnnotation")
      case name :: Nil =>
        (name, "")
      case "" :: name :: Nil => // this case handles a filename that begins with dot and has no suffix
        ("." + name, "")
      case other => {
        if (other.last.indexOf(File.separator) != -1) {
          (fileName, "")
        } else {
          (other.reverse.tail.reverse.mkString("."), "." + other.last)
        }
      }
    }
  }

  def getPrefix: String =
    prefix + originalMemoryNameOpt.map(n => target.name.drop(n.length)).getOrElse("")
  def getSuffix:   String = suffix
  def getFileName: String = getPrefix + getSuffix

  def duplicate(newNamed: Named): LoadMemoryAnnotation = {
    newNamed match {
      case componentName: ComponentName =>
        this.copy(target = componentName, originalMemoryNameOpt = Some(target.name))
      case _ =>
        throw new Exception(s"Cannot annotate anything but a memory, invalid target ${newNamed.serialize}")
    }
  }
}
