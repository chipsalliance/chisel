// See LICENSE for license details.

package firrtl.transforms

import java.io.{File, FileNotFoundException, FileOutputStream, PrintWriter}

import firrtl._
import firrtl.annotations.{Annotation, ModuleName}

import scala.collection.mutable.ArrayBuffer


trait BlackBoxSource {
  def serialize: String
  def name: String
}

object BlackBoxSource {
  val MaxFields = 3

  def parse(s: String): Option[BlackBoxSource] = {
    s.split("\n", MaxFields).toList  match {
      case "resource" :: id ::  _ => Some(BlackBoxResource(id))
      case "inline" :: name :: text :: _ => Some(BlackBoxInline(name, text))
      case "targetDir" :: targetDir :: _ => Some(BlackBoxTargetDir(targetDir))
      case _ => throw new FIRRTLException(s"Error: Bad BlackBox annotations $s")
    }
  }
}

case class BlackBoxTargetDir(targetDir: String) extends BlackBoxSource {
  def serialize: String = s"targetDir\n$targetDir"
  def name: String = targetDir
}

case class BlackBoxResource(resourceId: String) extends BlackBoxSource {
  def serialize: String = s"resource\n$resourceId"
  def name: String = resourceId.split("/").last
}

case class BlackBoxInline(name: String, text: String) extends BlackBoxSource {
  def serialize: String = s"inline\n$name\n$text"
}

object BlackBoxSourceAnnotation {
  def apply(targetDir: ModuleName, value: String): Annotation = {
    assert(BlackBoxSource.parse(value).isDefined)
    Annotation(targetDir, classOf[BlackBoxSourceHelper], value)
  }

  def unapply(a: Annotation): Option[(ModuleName, BlackBoxSource)] = a match {
    case Annotation(ModuleName(n, c), _, text) => Some((ModuleName(n, c), BlackBoxSource.parse(text).get))
    case _ => None
  }
}

/**
  * This transform handles the moving of verilator source for black boxes into the
  * target directory so that it can be accessed by verilator or other backend compilers
  * While parsing it's annotations it looks for a BlackBoxTargetDir annotation that
  * will set the directory where the verilog will be written.  This annotation is typically be
  * set by the execution harness, or directly in the tests
  */
class BlackBoxSourceHelper extends firrtl.Transform {
  private var targetDir: File = new File(".")
  private val fileList = new ArrayBuffer[String]

  override def inputForm: CircuitForm = LowForm
  override def outputForm: CircuitForm = LowForm

  /**
    * parse the annotations and convert the generic annotations to specific information
    * required to find the verilog
    * @note Side effect is that while converting a magic target dir annotation is found and sets the target
    * @param annos a list of generic annotations for this transform
    * @return
    */
  def getSources(annos: Seq[Annotation]): Seq[BlackBoxSource] = {
    annos.flatMap { anno => BlackBoxSource.parse(anno.value) }
      .flatMap {
        case BlackBoxTargetDir(dest) =>
          targetDir = new File(dest)
          if(! targetDir.exists()) { FileUtils.makeDirectory(targetDir.getAbsolutePath) }
          None
        case b: BlackBoxSource => Some(b)
        case _ => None
      }
      .sortBy(a => a.name)
      .distinct
  }

  /**
    * write the verilog source for each annotation to the target directory
    * @note the state is not changed by this transform
    * @param state Input Firrtl AST
    * @return A transformed Firrtl AST
    */
  override def execute(state: CircuitState): CircuitState = {
    val resultState = getMyAnnotations(state) match {
      case Nil => state
      case myAnnotations =>
        val sources = getSources(myAnnotations)
        sources.foreach {
          case BlackBoxResource(resourceId) =>
            val name = resourceId.split("/").last
            val outFile = new File(targetDir, name)
            BlackBoxSourceHelper.copyResourceToFile(resourceId,outFile)
            fileList += outFile.getAbsolutePath
          case BlackBoxInline(name, text) =>
            val outFile = new File(targetDir, name)
            val writer = new PrintWriter(outFile)
            writer.write(text)
            writer.close()
            fileList += outFile.getAbsolutePath
          case _ =>
        }
        state
    }
    // If we have BlackBoxes, generate the helper file.
    // If we don't, make sure it doesn't exist or we'll confuse downstream processing
    //  that triggers behavior on the existence of the file
    val helperFile = new File(targetDir, BlackBoxSourceHelper.FileListName)
    if(fileList.nonEmpty) {
      val writer = new PrintWriter(helperFile)
      writer.write(fileList.map { fileName => s"-v $fileName" }.mkString("\n"))
      writer.close()
    } else {
      helperFile.delete()
    }

    resultState
  }
}

object BlackBoxSourceHelper {
  val FileListName = "black_box_verilog_files.f"
  /**
    * finds the named resource and writes into the directory
    * @param name the name of the resource
    * @param file the file to write it into
    */
  def copyResourceToFile(name: String, file: File) {
    val in = getClass.getResourceAsStream(name)
    if (in == null) {
      throw new FileNotFoundException(s"Resource '$name'")
    }
    val out = new FileOutputStream(file)
    Iterator.continually(in.read).takeWhile(-1 != _).foreach(out.write)
    out.close()
  }

}
