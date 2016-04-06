// See LICENSE for license details.

package Chisel

@deprecated("FileSystemUtilities doesn't exist in chisel3", "3.0.0")
trait FileSystemUtilities {
  def createOutputFile(name: String): java.io.FileWriter  = {
    new java.io.FileWriter(Driver.targetDir + "/" + name)
  }
}
