// SPDX-License-Identifier: Apache-2.0

package firrtl

object FileUtils {

  /** Read a text file and return it as a Seq of strings
    * Closes the file after read to avoid dangling file handles
    *
    * @param fileName The file to read
    */
  def getLines(fileName: String): Seq[String] = getLines(getPath(fileName))

  /** Read a text file and return it as  a Seq of strings
    * Closes the file after read to avoid dangling file handles
    * @param file an os.Path to be read
    */
  def getLines(file: os.Path): Seq[String] = os.read.lines(file)

  /** Read a text file and return it as  a single string
    * Closes the file after read to avoid dangling file handles
    *
    * @param fileName The file to read
    */
  def getText(fileName: String): String = getText(getPath(fileName))

  /** Read a text file and return it as  a single string
    * Closes the file after read to avoid dangling file handles
    *
    * @param file an os.Path to be read
    */
  def getText(file: os.Path): String = os.read(file)

  /** Get os.Path from String
    * @param pathName an absolute or relative path string
    */
  def getPath(pathName: String): os.Path = os.FilePath(pathName) match {
    case path: os.Path    => path
    case sub:  os.SubPath => os.pwd / sub
    case rel:  os.RelPath => os.pwd / rel
  }
}
