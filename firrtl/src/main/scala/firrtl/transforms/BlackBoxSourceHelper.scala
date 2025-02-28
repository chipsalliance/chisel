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

/** Exception indicating that a blackbox wasn't found
  * @param fileName the name of the BlackBox file (only used for error message generation)
  * @param e an underlying exception that generated this
  */
class BlackBoxNotFoundException(fileName: String, message: String)
    extends FirrtlUserException(
      s"BlackBox '$fileName' not found. Did you misspell it? Is it in src/{main,test}/resources?\n$message"
    )
