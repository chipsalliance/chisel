package chisel3.simulator

import java.io.File
import chisel3.layer.Layer

/** Utilities for enabling and disabling Chisel layers */
object LayerControl {

  /** The type of all layer control variations */
  sealed trait Type {

    /** Return true if a file should be included in the build.  This will always
      * return true for any non-layer file.
      * @param file the file to check
      */
    final def filter(file: File): Boolean = file.getName match {
      case nonLayer if !nonLayer.startsWith("layers-") => true
      case layer                                       => shouldEnable(layer)
    }

    /** Return true if a layer should be included in the build.
      * @param layerFilename the filename of a layer
      */
    protected def shouldEnable(layerFilename: String): Boolean
  }

  /** Enable all layers */
  final case object EnableAll extends Type {
    override protected def shouldEnable(layerFilename: String) = true
  }

  /** Enable only the specified layers
    *
    * Nested layers should use a `.` as a delimiter.
    *
    * @param layers a variadic list of layer names
    */
  final case class Enable(layers: Layer*) extends Type {
    private val _shouldEnable: String => Boolean = {
      layers match {
        case Nil => _ => false
        case _ =>
          val layersRe = layers.map(_.fullName.split("\\.").mkString("-")).mkString("|")
          val re = s"^layers-\\w+-($layersRe)\\.sv$$".r
          re.matches(_)
      }
    }
    override protected def shouldEnable(filename: String) = _shouldEnable(filename)
  }

  /** Disables all layers.  This is the same as `Enable()`. */
  val DisableAll = Enable()

}
