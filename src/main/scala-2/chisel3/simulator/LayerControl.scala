package chisel3.simulator

import java.io.File
import chisel3.RawModule
import chisel3.layer.{ABI, Layer}
import chisel3.simulator.ElaboratedModule
import chisel3.stage.DesignAnnotation
import svsim.CommonCompilationSettings.VerilogPreprocessorDefine

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

    /** Return the layers that should be enabled in a circuit.  The layers must exist in the design.
      *
      * @param design an Annotation that contains an elaborated design used to check that the requested layers exist
      * @return all layers that should be enabled
      * @throws IllegalArgumentException if the requested layers
      */
    protected def getLayerSubset(module: ElaboratedModule[_]): Seq[Layer]

    /** Return the preprocessor defines that should be set to enable the layers of
      * this `LayerControl.Type`.
      *
      * This requires passing an elaborated module in order to know what layers
      * exist in the design.
      *
      * @param module an elaborated module
      * @return preprocessor defines to control the enabling of these layers
      */
    final def preprocessorDefines(
      module: ElaboratedModule[_ <: RawModule]
    ): Seq[VerilogPreprocessorDefine] = getLayerSubset(module).flatMap {
      case layer =>
        layer.config.abi match {
          case abi: chisel3.layer.ABI.PreprocessorDefine.type =>
            Some(VerilogPreprocessorDefine(abi.toMacroIdentifier(layer, module.wrapped.circuitName)))
          case _ => None
        }
    }

    /** Return a partial function that will return true if a file should be included
      * in the build to enable a layer.  This partial function is not defined if
      * the file is not a layer file.
      *
      * @param module an elaborated module
      * @return a partial function to test if layer files should be included
      */
    final def shouldIncludeFile(module: ElaboratedModule[_ <: RawModule]): PartialFunction[File, Boolean] = {
      val layerFilenames: Seq[String] = getLayerSubset(module).flatMap {
        case layer =>
          layer.config.abi match {
            case abi: chisel3.layer.ABI.FileInclude.type =>
              Some(abi.toFilename(layer, module.wrapped.circuitName))
            case _ => None
          }
      }

      {
        case a if a.getName().startsWith("layers-") =>
          layerFilenames.mkString("|").r.matches(a.getName())
      }
    }

  }

  /** Enable all layers */
  final case object EnableAll extends Type {
    override protected def shouldEnable(layerFilename: String) = true

    override protected def getLayerSubset(module: ElaboratedModule[_]): Seq[Layer] = module.layers
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

    override protected def getLayerSubset(module: ElaboratedModule[_]): Seq[Layer] = {
      val definedLayers = module.layers
      layers.foreach { layer =>
        require(
          definedLayers.contains(layer),
          s"""cannot enable layer '${layer.fullName}' as it is not one of the defined layers: ${definedLayers.map(
            _.fullName
          )}"""
        )
      }
      layers
    }
  }

  /** Disables all layers.  This is the same as `Enable()`. */
  val DisableAll = Enable()

}
