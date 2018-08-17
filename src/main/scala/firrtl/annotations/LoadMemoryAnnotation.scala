// See LICENSE for license details.

package firrtl.annotations

/**
  * Enumeration of the two types of readmem statements available in verilog
  */
object MemoryLoadFileType extends Enumeration {
  type FileType = Value

  val Hex:    Value = Value("h")
  val Binary: Value = Value("b")
}

/**
  * Firrtl implementation for load memory
  * @param target        memory to load
  * @param fileName      name of input file
  * @param hexOrBinary   use $readmemh or $readmemb
  */
case class LoadMemoryAnnotation(
  target: ComponentName,
  fileName: String,
  hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex,
  originalMemoryNameOpt: Option[String] = None
) extends SingleTargetAnnotation[Named] {

  val (prefix, suffix) = {
    fileName.split("""\.""").toList match {
      case Nil =>
        throw new Exception(s"empty filename not allowed in LoadMemoryAnnotation")
      case name :: Nil =>
        (name, "")
      case "" :: name :: Nil => // this case handles a filename that begins with dot and has no suffix
        ("." + name, "")
      case other =>
        (other.reverse.tail.reverse.mkString("."), "." + other.last)
    }
  }

  def getFileName: String = {
    originalMemoryNameOpt match {
      case Some(originalMemoryName) =>
        if(target.name == originalMemoryName) {
          prefix + suffix
        }
        else {
          prefix + target.name.drop(originalMemoryName.length) + suffix
        }
      case _ =>
        fileName
    }
  }

  def getSuffix: String = suffix

  def duplicate(newNamed: Named): LoadMemoryAnnotation = {
    newNamed match {
      case componentName: ComponentName =>
        this.copy(target = componentName, originalMemoryNameOpt = Some(target.name))
      case _ =>
        throw new Exception(s"Cannot annotate anything but a memory, invalid target ${newNamed.serialize}")
    }
  }
}


