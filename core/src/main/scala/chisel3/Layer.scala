// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.Data.ProbeInfo
import chisel3.experimental.{requireIsHardware, SourceInfo, UnlocatableSourceInfo}
import chisel3.internal.{Builder, HasId}
import chisel3.internal.firrtl.ir.{LayerBlock, Node}
import chisel3.reflect.DataMirror
import chisel3.util.simpleClassName
import java.nio.file.{Path, Paths}
import scala.annotation.tailrec
import scala.collection.mutable.{ArrayBuffer, LinkedHashSet}

/** This object contains Chisel language features for creating layers.  Layers
  * are collections of hardware that are not always present in the circuit.
  * Layers are intended to be used to hold verification or debug code.
  */
object layer {

  /** Enumerations of different optional layer conventions.  A layer convention
    * says how a given layer should be lowered to Verilog.
    */
  @deprecated("`Convention` is being removed in favor of `LayerConfig`", "Chisel 7.0.0")
  object Convention {
    sealed trait Type

    /** The layer should be lowered to a SystemVerilog `bind`. */
    final case object Bind extends Type
  }

  /** Enumeration of different application binary interfaces (ABIs) for how to
    * enable layers.  These are implementations of the FIRRTL ABI Specification.
    */
  object ABI {
    sealed trait Type

    /** An ABI that is implemented as a file included during Verilog elabortion. */
    final case object FileInclude extends Type {

      /** Retun the file name of
        */
      def toFilename(layer: Layer, circuitName: String): String = {
        (s"layers-$circuitName" +: layer.layerSeq.map(_.name)).mkString("-") + ".sv"
      }

    }

    /** An ABI that requires a preprocessor macro identifier to be defined during Verilog elaboration.
      */
    final case object PreprocessorDefine extends Type {

      /** Return the macro identifier that should be defined. */
      def toMacroIdentifier(layer: Layer, circuitName: String): String = {
        ("layer" +: layer.layerSeq.map(_.name)).mkString("$")
      }

    }

    /** A dummy ABI for the root LayerConfig.  This should be unused otherwise. */
    final case object Root extends Type
  }

  sealed trait OutputDirBehavior
  final case object DefaultOutputDir extends OutputDirBehavior
  final case object NoOutputDir extends OutputDirBehavior
  final case class CustomOutputDir(path: Path) extends OutputDirBehavior

  sealed trait LayerConfig {
    def abi: ABI.Type
  }
  object LayerConfig {
    final case class Extract(outputDirBehavior: OutputDirBehavior = DefaultOutputDir) extends LayerConfig {
      override val abi: ABI.Type = ABI.FileInclude
    }
    final case object Inline extends LayerConfig {
      override val abi: ABI.Type = ABI.PreprocessorDefine
    }
    private[chisel3] final case object Root extends LayerConfig {
      override val abi: ABI.Type = ABI.Root
    }
  }

  /** A layer declaration.
    *
    * @param convention how this layer should be lowered
    * @param outputDirBehavior an optional user-provided output directory for this layer
    * @param _parent the parent layer, if any
    */
  abstract class Layer(
    val config: LayerConfig
  )(implicit _parent: Layer, _sourceInfo: SourceInfo) {
    self: Singleton =>

    @deprecated("`Convention` is being removed in favor of `LayerConfig`", "Chisel 7.0.0")
    def this(
      convention:        Convention.Type,
      outputDirBehavior: OutputDirBehavior = DefaultOutputDir
    )(
      implicit _parent: Layer,
      _sourceInfo:      SourceInfo
    ) = this(LayerConfig.Extract(outputDirBehavior))

    /** This establishes a new implicit val for any nested layers. */
    protected final implicit val thiz: Layer = this

    private[chisel3] def parent: Layer = _parent

    private[chisel3] def sourceInfo: SourceInfo = _sourceInfo

    private[chisel3] def name: String = simpleClassName(this.getClass())

    private[chisel3] val fullName: String = parent match {
      case null       => "<root>"
      case Layer.Root => name
      case _          => s"${parent.fullName}.$name"
    }

    /** The output directory of this layer.
      *
      * The output directory of a layer serves as a hint to the toolchain,
      * specifying where files related to the layer (such as a bindfile, in
      * verilog) should be output. If a layer has no output directory, then files
      * related to this layer will be placed in the default output directory.
      *
      * Unless overridden, the outputDir's name matches the name of the layer,
      * and is located under the parent layer's directory.
      */
    private[chisel3] final def outputDir: Option[Path] = {
      config match {
        case LayerConfig.Root   => None
        case LayerConfig.Inline => None
        case LayerConfig.Extract(outputDirBehavior) =>
          outputDirBehavior match {
            case NoOutputDir          => None
            case CustomOutputDir(dir) => Some(dir)
            case DefaultOutputDir     => Some(defaultOutputDir)
          }
      }
    }

    private final def defaultOutputDir: Path = parentOutputDir match {
      case None      => Paths.get(name)
      case Some(dir) => dir.resolve(name)
    }

    private final def parentOutputDir: Option[Path] = {
      @tailrec
      def outputDirOf(layer: Layer): Option[Path] = layer.config match {
        case LayerConfig.Extract(_) => layer.outputDir
        case LayerConfig.Inline     => outputDirOf(layer.parent)
        case LayerConfig.Root       => None
      }
      outputDirOf(parent)
    }

    @tailrec
    final private[chisel3] def canWriteTo(that: Layer): Boolean = that match {
      case null              => false
      case _ if this == that => true
      case _                 => this.canWriteTo(that.parent)
    }

    /** Return a list containg this layer and all its parents, excluding the root
      * layer.  The deepest layer (this layer) is the first element in the list.
      *
      * @return a sequence of all layers
      */
    private[chisel3] def layerSeq: List[Layer] = {
      def rec(current: Layer, acc: List[Layer]): List[Layer] = current match {
        case Layer.root => acc
        case _          => rec(current.parent, current :: acc)
      }
      rec(this, Nil)
    }
  }

  object Layer {
    private[chisel3] final case object Root extends Layer(LayerConfig.Root)(null, UnlocatableSourceInfo)
    implicit val root: Layer = Root
  }

  /** Add a layer and all of its parents to the Builder.  This lets the Chisel
    * know that this layer should be emitted into FIRRTL text.
    *
    * This API can be used to guarantee that a design will always have certain
    * layers defined.  By default, layers are only included in the FIRRTL text
    * if they have layer block users.
    */
  def addLayer(layer: Layer) = {
    var currentLayer: Layer = layer
    while (currentLayer != Layer.Root && !Builder.layers.contains(currentLayer)) {
      val layer = currentLayer
      val parent = layer.parent

      Builder.layers += layer
      currentLayer = parent
    }

    // If this API is used in a BlackBox, then modify it's `knownLayers` member.
    Builder.currentModule.map {
      case module: internal.BaseBlackBox => module.addKnownLayer(layer)
      case _ =>
    }
  }

  /** A type class that describes how to post-process the return value from a layer block. */
  sealed trait BlockReturnHandler[A] {

    /** The type that will be returned by this handler. */
    type R

    /** Do not post-process the return value.  This is what should happen if the
      * code early exits, i.e., if no layer block is created.
      */
    private[chisel3] def identity(result: A): R

    /** Post-process the return value.
      *
      * @param placeholder the location above the layer block where commands can be inserted
      * @param layerBlock the current layer block
      * @param result the return value of the layer block
      * @param sourceInfo source locator information
      * @return the post-processed result
      */
    private[chisel3] def apply(placeholder: Placeholder, layerBlock: LayerBlock, result: A)(
      implicit sourceInfo: SourceInfo
    ): R
  }

  object BlockReturnHandler {

    type Aux[A, B] = BlockReturnHandler[A] { type R = B }

    /** Return a [[BlockReturnHandler]] that will always return [[Unit]]. */
    def unit[A]: Aux[A, Unit] = new BlockReturnHandler[A] {

      override type R = Unit

      override private[chisel3] def identity(result: A): R = ()

      override private[chisel3] def apply(
        placeholder: Placeholder,
        layerBlock:  LayerBlock,
        result:      A
      )(
        implicit sourceInfo: SourceInfo
      ): R = {
        ()
      }

    }

    /** Return a [[BlockReturnHandler]] that, if a layer block is created, will
      * create a [[Wire]] of layer-colored probe type _above_ the layer block
      * and [[probe.define]] this at the end of the layer block.  If no
      * layer-block is created, then the result is pass-through.
      */
    implicit def layerColoredWire[A <: Data]: Aux[A, A] = new BlockReturnHandler[A] {

      override type R = A

      override private[chisel3] def identity(result: A): R = result

      override private[chisel3] def apply(
        placeholder: Placeholder,
        layerBlock:  LayerBlock,
        result:      A
      )(
        implicit sourceInfo: SourceInfo
      ): R = {
        // Don't allow returning non-hardware types.  This avoids problems like
        // returning a probe type which we can't handle.
        requireIsHardware(result)

        // The result wire is either a new layer-colored probe wire or an
        // existing layer-colored probe wire forwarded up.  If it is the former,
        // the wire is colored with the current layer.  For the latter, the wire
        // needs to have a compatible color.  If it has one, use it.  If it
        // needs a color, set it on the wire.
        val layerColoredWire = placeholder.append {
          result.probeInfo match {
            case None                     => Wire(probe.Probe(chiselTypeOf(result), layerBlock.layer))
            case Some(ProbeInfo(_, None)) => Wire(probe.Probe(result.cloneType, layerBlock.layer))
            case Some(ProbeInfo(_, Some(color))) =>
              if (!layerBlock.layer.canWriteTo(color))
                Builder.error(
                  s"cannot return probe of color '${color.fullName}' from a layer block associated with layer '${layerBlock.layer.fullName}'"
                )
              Wire(chiselTypeOf(result))
          }
        }

        // Similarly, if the result is a probe, then we forward it directly.  If
        // it is a non-probe, then we need to probe it first.
        Builder.forcedUserModule.withRegion(layerBlock.region) {
          val source = DataMirror.hasProbeTypeModifier(result) match {
            case true  => result
            case false => probe.ProbeValue(result)
          }
          probe.define(layerColoredWire, source)
        }

        // Return the layer-colored wire _above_ the layer block.
        layerColoredWire
      }

    }

  }

  /** Create a new layer block.  This is hardware that will be enabled
    * collectively when the layer is enabled.
    *
    * This function automatically creates parent layers as the layer of the
    * current layerblock is an ancestor of the desired layer.  The ancestor may
    * be the current layer which causes no layer block to be created.  (This is
    * not a _proper_ ancestor requirement.)
    *
    * By default, the return of the layer block will be either a subtype of
    * [[Data]] or [[Unit]], depending on the `thunk` provided.  If the `thunk`
    * (the hardware that should be constructed inside the layer block) returns a
    * [[Data]], then this will either return a [[Wire]] of layer-colored
    * [[Probe]] type if a layer block was created or the underlying [[Data]] if
    * no layer block was created.  If the `thunk` returns anything else, this
    * will return [[Unit]].  This is controlled by the implicit argument `tc`
    * and may be customized by advanced users to do other things.
    *
    * @param layer the layer this block is associated with
    * @param skipIfAlreadyInBlock if true, then this will not create a layer if
    * this `block` is already inside another layerblock
    * @param skipIfLayersEnabled if true, then this will not create a layer if
    * any layers have been enabled for the current module block if already
    * inside a layer block
    * @param thunk the Chisel code that goes into the layer block
    * @param sourceInfo a source locator
    * @throws java.lang.IllegalArgumentException if the layer of the currnet
    * layerblock is not an ancestor of the desired layer
    * @return either a subtype of [[Data]] or [[Unit]] depending on the `thunk`
    * return type
    */
  def block[A](
    layer:                Layer,
    skipIfAlreadyInBlock: Boolean = false,
    skipIfLayersEnabled:  Boolean = false
  )(thunk: => A)(
    implicit tc: BlockReturnHandler[A] = BlockReturnHandler.unit[A],
    sourceInfo:  SourceInfo
  ): tc.R = {
    // Do nothing if we are already in a layer block and are not supposed to
    // create new layer blocks.
    if (
      skipIfAlreadyInBlock && Builder.layerStack.size > 1 || skipIfLayersEnabled && Builder.enabledLayers.nonEmpty || Builder.elideLayerBlocks
    ) {
      return tc.identity(thunk)
    }

    val _layer = Builder.layerMap.getOrElse(layer, layer)
    var layersToCreate = List.empty[Layer]
    var currentLayer = _layer
    while (currentLayer != Builder.layerStack.head && currentLayer != Layer.Root) {
      layersToCreate = currentLayer :: layersToCreate
      currentLayer = currentLayer.parent
    }
    require(
      currentLayer != Layer.Root || Builder.layerStack.head == Layer.Root,
      s"a layerblock associated with layer '${_layer.fullName}' cannot be created under a layerblock of non-ancestor layer '${Builder.layerStack.head.fullName}'"
    )

    if (layersToCreate.isEmpty)
      return tc.identity(thunk)

    addLayer(_layer)

    // Save the append point _before_ the layer block so that we can insert a
    // layer-colored wire once the `thunk` executes.
    val beforeLayerBlock = new Placeholder

    // Track the current layer block.  When this is used, this will be the
    // innermost layer block that will be created.  This is guaranteed to be
    // non-null as long as `layersToCreate` is not empty.
    var layerBlock: LayerBlock = null

    // Recursively create any necessary layers.  There are two cases:
    //
    // 1. There are no layers left to create.  Run the thunk, create the
    //    layer-colored wire, and define it.
    // 2. There are layers left to create.  Create the next layer and recurse.
    def createLayers(layers: List[Layer])(thunk: => A): A = layers match {
      case Nil => thunk
      case head :: tail =>
        layerBlock = Builder.pushCommand(new LayerBlock(sourceInfo, head))
        Builder.layerStack = head :: Builder.layerStack
        val result = Builder.forcedUserModule.withRegion(layerBlock.region)(createLayers(tail)(thunk))
        Builder.layerStack = Builder.layerStack.tail
        result
    }

    val result = createLayers(layersToCreate)(thunk)
    return tc.apply(beforeLayerBlock, layerBlock, result)
  }

  /** API that will cause any calls to `block` in the `thunk` to not create new
    * layer blocks.
    *
    * This is an advanced, library-writer's API that is not intended for broad
    * use.  You may consider using this if you are writing an API which
    * automatically puts code into layer blocks and you want to provide a way
    * for a user to completely opt out of this.
    *
    * @param thunk the Chisel code that should not go into a layer block
    */
  def elideBlocks[A](thunk: => A): A = {
    val oldElideLayerBlocks = Builder.elideLayerBlocks
    Builder.elideLayerBlocks = true
    val result = thunk
    Builder.elideLayerBlocks = oldElideLayerBlocks
    return result
  }

  /** Call this function from within a `Module` body to enable this layer globally for that module. */
  final def enable(layer: Layer): Unit = layer match {
    case Layer.Root =>
    case _ =>
      addLayer(layer)
      Builder.enabledLayers += layer
  }

}
