// SPDX-License-Identifier: Apache-2.0

package firrtl.transforms

import java.io.{File, FileNotFoundException, FileOutputStream}

import firrtl._
import firrtl.annotations._

sealed trait BlackBoxHelperAnno extends Annotation

case class BlackBoxTargetDirAnno(targetDir: String) extends BlackBoxHelperAnno with NoTargetAnnotation {
  override def serialize: String = s"targetDir\n$targetDir"
}

case class BlackBoxInlineAnno(target: ModuleName, name: String, text: String)
    extends BlackBoxHelperAnno
    with SingleTargetAnnotation[ModuleName] {
  def duplicate(n: ModuleName) = this.copy(target = n)
  override def serialize: String = s"inline\n$name\n$text"
}

case class BlackBoxPathAnno(target: ModuleName, path: String)
    extends BlackBoxHelperAnno
    with SingleTargetAnnotation[ModuleName] {
  def duplicate(n: ModuleName) = this.copy(target = n)
  override def serialize: String = s"path\n$path"
}

case class BlackBoxResourceFileNameAnno(resourceFileName: String) extends BlackBoxHelperAnno with NoTargetAnnotation {
  override def serialize: String = s"resourceFileName\n$resourceFileName"
}

/** Exception indicating that a blackbox wasn't found
  * @param fileName the name of the BlackBox file (only used for error message generation)
  * @param e an underlying exception that generated this
  */
class BlackBoxNotFoundException(fileName: String, message: String)
    extends FirrtlUserException(
      s"BlackBox '$fileName' not found. Did you misspell it? Is it in src/{main,test}/resources?\n$message"
    )

object BlackBoxSourceHelper {

  /** Safely access a file converting [[FileNotFoundException]]s and [[NullPointerException]]s into
    * [[BlackBoxNotFoundException]]s
    * @param fileName the name of the file to be accessed (only used for error message generation)
    * @param code some code to run
    */
  private def safeFile[A](fileName: String)(code: => A) = try { code }
  catch {
    case e @ (_: FileNotFoundException | _: NullPointerException) =>
      throw new BlackBoxNotFoundException(fileName, e.getMessage)
  }

  /**
    * finds the named resource and writes into the directory
    * @param name the name of the resource
    * @param dir the directory in which to write the file
    * @return the closed File object
    */
  def writeResourceToDirectory(name: String, dir: File): File = {
    val fileName = name.split("/").last
    val outFile = new File(dir, fileName)
    copyResourceToFile(name, outFile)
    outFile
  }

  /**
    * finds the named resource and writes into the directory
    * @param name the name of the resource
    * @param file the file to write it into
    * @throws BlackBoxNotFoundException if the requested resource does not exist
    */
  def copyResourceToFile(name: String, file: File): Unit = {
    val in = getClass.getResourceAsStream(name)
    val out = new FileOutputStream(file)
    safeFile(name)(Iterator.continually(in.read).takeWhile(-1 != _).foreach(out.write))
    out.close()
  }

  val defaultFileListName = "firrtl_black_box_resource_files.f"

}
