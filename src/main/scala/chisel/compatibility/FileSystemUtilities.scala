// See LICENSE for license details.

package chisel.compatibility

import chisel._

@deprecated("FileSystemUtilities doesn't exist in chisel3", "3.0.0")
trait FileSystemUtilities {
  def createOutputFile(name: String): java.io.FileWriter  = {
    new java.io.FileWriter(Driver.targetDir + "/" + name)
  }
}
