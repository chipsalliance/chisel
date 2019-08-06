-----------------------------------------------------
chiselFrontend/src/main/scala/chisel3/internal/firrtl
-----------------------------------------------------

.. toctree::


IR.scala
--------
.. chisel:attr:: case class PrimOp(val name: String)


.. chisel:attr:: object PrimOp


.. chisel:attr:: abstract class Arg


.. chisel:attr:: case class Node(id: HasId) extends Arg


.. chisel:attr:: abstract class LitArg(val num: BigInt, widthArg: Width) extends Arg


.. chisel:attr:: case class ILit(n: BigInt) extends Arg


.. chisel:attr:: case class ULit(n: BigInt, w: Width) extends LitArg(n, w)


.. chisel:attr:: case class SLit(n: BigInt, w: Width) extends LitArg(n, w)


.. chisel:attr:: case class FPLit(n: BigInt, w: Width, binaryPoint: BinaryPoint) extends LitArg(n, w)


.. chisel:attr:: case class Ref(name: String) extends Arg case class ModuleIO(mod: BaseModule, name: String) extends Arg


.. chisel:attr:: case class ModuleIO(mod: BaseModule, name: String) extends Arg


.. chisel:attr:: case class Slot(imm: Node, name: String) extends Arg


.. chisel:attr:: case class Index(imm: Arg, value: Arg) extends Arg


.. chisel:attr:: sealed trait Bound sealed trait NumericBound[T] extends Bound


.. chisel:attr:: sealed trait NumericBound[T] extends Bound


.. chisel:attr:: sealed case class Open[T](value: T) extends NumericBound[T] sealed case class Closed[T](value: T) extends NumericBound[T]  sealed trait Range


.. chisel:attr:: sealed case class Closed[T](value: T) extends NumericBound[T]  sealed trait Range


.. chisel:attr:: sealed trait Range


.. chisel:attr:: sealed trait KnownIntRange extends Range


.. chisel:attr:: sealed case class KnownUIntRange(min: NumericBound[Int], max: NumericBound[Int]) extends KnownIntRange


.. chisel:attr:: sealed case class KnownSIntRange(min: NumericBound[Int], max: NumericBound[Int]) extends KnownIntRange


.. chisel:attr:: object Width


.. chisel:attr:: sealed abstract class Width


.. chisel:attr:: sealed case class UnknownWidth() extends Width


.. chisel:attr:: sealed case class KnownWidth(value: Int) extends Width


.. chisel:attr:: object BinaryPoint


.. chisel:attr:: sealed abstract class BinaryPoint


.. chisel:attr:: case object UnknownBinaryPoint extends BinaryPoint


.. chisel:attr:: sealed case class KnownBinaryPoint(value: Int) extends BinaryPoint


.. chisel:attr:: sealed abstract class MemPortDirection(name: String)


.. chisel:attr:: object MemPortDirection


.. chisel:attr:: abstract class Command


.. chisel:attr:: abstract class Definition extends Command


.. chisel:attr:: case class DefPrim[T <: Data](sourceInfo: SourceInfo, id: T, op: PrimOp, args: Arg*) extends Definition case class DefInvalid(sourceInfo: SourceInfo, arg: Arg) extends Command case class DefWire(sourceInfo: SourceInfo, id: Data) extends Definition


.. chisel:attr:: case class DefInvalid(sourceInfo: SourceInfo, arg: Arg) extends Command case class DefWire(sourceInfo: SourceInfo, id: Data) extends Definition case class DefReg(sourceInfo: SourceInfo, id: Data, clock: Arg) extends Definition


.. chisel:attr:: case class DefWire(sourceInfo: SourceInfo, id: Data) extends Definition case class DefReg(sourceInfo: SourceInfo, id: Data, clock: Arg) extends Definition case class DefRegInit(sourceInfo: SourceInfo, id: Data, clock: Arg, reset: Arg, init: Arg) extends Definition


.. chisel:attr:: case class DefReg(sourceInfo: SourceInfo, id: Data, clock: Arg) extends Definition case class DefRegInit(sourceInfo: SourceInfo, id: Data, clock: Arg, reset: Arg, init: Arg) extends Definition case class DefMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt) extends Definition


.. chisel:attr:: case class DefRegInit(sourceInfo: SourceInfo, id: Data, clock: Arg, reset: Arg, init: Arg) extends Definition case class DefMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt) extends Definition case class DefSeqMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt) extends Definition


.. chisel:attr:: case class DefMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt) extends Definition case class DefSeqMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt) extends Definition case class DefMemPort[T <: Data](sourceInfo: SourceInfo, id: T, source: Node, dir: MemPortDirection, index: Arg, clock: Arg) extends Definition


.. chisel:attr:: case class DefSeqMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt) extends Definition case class DefMemPort[T <: Data](sourceInfo: SourceInfo, id: T, source: Node, dir: MemPortDirection, index: Arg, clock: Arg) extends Definition case class DefInstance(sourceInfo: SourceInfo, id: BaseModule, ports: Seq[Port]) extends Definition


.. chisel:attr:: case class DefMemPort[T <: Data](sourceInfo: SourceInfo, id: T, source: Node, dir: MemPortDirection, index: Arg, clock: Arg) extends Definition case class DefInstance(sourceInfo: SourceInfo, id: BaseModule, ports: Seq[Port]) extends Definition case class WhenBegin(sourceInfo: SourceInfo, pred: Arg) extends Command


.. chisel:attr:: case class DefInstance(sourceInfo: SourceInfo, id: BaseModule, ports: Seq[Port]) extends Definition case class WhenBegin(sourceInfo: SourceInfo, pred: Arg) extends Command case class WhenEnd(sourceInfo: SourceInfo, firrtlDepth: Int, hasAlt: Boolean


.. chisel:attr:: case class WhenBegin(sourceInfo: SourceInfo, pred: Arg) extends Command case class WhenEnd(sourceInfo: SourceInfo, firrtlDepth: Int, hasAlt: Boolean


.. chisel:attr:: case class WhenEnd(sourceInfo: SourceInfo, firrtlDepth: Int, hasAlt: Boolean


.. chisel:attr:: case class AltBegin(sourceInfo: SourceInfo) extends Command case class OtherwiseEnd(sourceInfo: SourceInfo, firrtlDepth: Int) extends Command case class Connect(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command


.. chisel:attr:: case class OtherwiseEnd(sourceInfo: SourceInfo, firrtlDepth: Int) extends Command case class Connect(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command case class BulkConnect(sourceInfo: SourceInfo, loc1: Node, loc2: Node) extends Command


.. chisel:attr:: case class Connect(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command case class BulkConnect(sourceInfo: SourceInfo, loc1: Node, loc2: Node) extends Command case class Attach(sourceInfo: SourceInfo, locs: Seq[Node]) extends Command


.. chisel:attr:: case class BulkConnect(sourceInfo: SourceInfo, loc1: Node, loc2: Node) extends Command case class Attach(sourceInfo: SourceInfo, locs: Seq[Node]) extends Command case class ConnectInit(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command


.. chisel:attr:: case class Attach(sourceInfo: SourceInfo, locs: Seq[Node]) extends Command case class ConnectInit(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command case class Stop(sourceInfo: SourceInfo, clock: Arg, ret: Int) extends Command


.. chisel:attr:: case class ConnectInit(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command case class Stop(sourceInfo: SourceInfo, clock: Arg, ret: Int) extends Command case class Port(id: Data, dir: SpecifiedDirection)


.. chisel:attr:: case class Stop(sourceInfo: SourceInfo, clock: Arg, ret: Int) extends Command case class Port(id: Data, dir: SpecifiedDirection) case class Printf(sourceInfo: SourceInfo, clock: Arg, pable: Printable) extends Command


.. chisel:attr:: case class Port(id: Data, dir: SpecifiedDirection) case class Printf(sourceInfo: SourceInfo, clock: Arg, pable: Printable) extends Command abstract class Component extends Arg


.. chisel:attr:: case class Printf(sourceInfo: SourceInfo, clock: Arg, pable: Printable) extends Command abstract class Component extends Arg


.. chisel:attr:: abstract class Component extends Arg


.. chisel:attr:: case class DefModule(id: RawModule, name: String, ports: Seq[Port], commands: Seq[Command]) extends Component case class DefBlackBox(id: BaseBlackBox, name: String, ports: Seq[Port], topDir: SpecifiedDirection, params: Map[String, Param]) extends Component  case class Circuit(name: String, components: Seq[Component], annotations: Seq[ChiselAnnotation]


.. chisel:attr:: case class DefBlackBox(id: BaseBlackBox, name: String, ports: Seq[Port], topDir: SpecifiedDirection, params: Map[String, Param]) extends Component  case class Circuit(name: String, components: Seq[Component], annotations: Seq[ChiselAnnotation]


.. chisel:attr:: case class Circuit(name: String, components: Seq[Component], annotations: Seq[ChiselAnnotation]


