// Private implicit classes and other utility functions for debugging

package firrtl

import Utils._

private object DebugUtils {

  implicit class DebugASTUtils(ast: AST) {
    // Is this actually any use?
    def preOrderTraversal(f: AST => Unit): Unit = {
      f(ast)
      ast match {
        case a: Block => a.stmts.foreach(_.preOrderTraversal(f))
        case a: Assert => a.pred.preOrderTraversal(f)
        case a: When => {
          a.pred.preOrderTraversal(f)
          a.conseq.preOrderTraversal(f)
          a.alt.preOrderTraversal(f)
        }
        case a: BulkConnect => {
          a.lhs.preOrderTraversal(f)
          a.rhs.preOrderTraversal(f)
        }
        case a: Connect => {
          a.lhs.preOrderTraversal(f)
          a.rhs.preOrderTraversal(f)
        }
        case a: OnReset => {
          a.lhs.preOrderTraversal(f)
          a.rhs.preOrderTraversal(f)
        }
        case a: DefAccessor => {
          a.dir.preOrderTraversal(f)
          a.source.preOrderTraversal(f)
          a.index.preOrderTraversal(f)
        }
        case a: DefPoison => a.tpe.preOrderTraversal(f)
        case a: DefNode => a.value.preOrderTraversal(f)
        case a: DefInst => a.module.preOrderTraversal(f)
        case a: DefMemory => {
          a.tpe.preOrderTraversal(f)
          a.clock.preOrderTraversal(f)
        }
        case a: DefReg => {
          a.tpe.preOrderTraversal(f)
          a.clock.preOrderTraversal(f)
          a.reset.preOrderTraversal(f)
        }
        case a: DefWire => a.tpe.preOrderTraversal(f)
        case a: Field => {
          a.dir.preOrderTraversal(f)
          a.tpe.preOrderTraversal(f)
        }
        case a: VectorType => a.tpe.preOrderTraversal(f)
        case a: BundleType => a.fields.foreach(_.preOrderTraversal(f))
        case a: Port => {
          a.dir.preOrderTraversal(f)
          a.tpe.preOrderTraversal(f)
        }
        case a: Module => {
          a.ports.foreach(_.preOrderTraversal(f))
          a.stmt.preOrderTraversal(f)
        }
        case a: Circuit => a.modules.foreach(_.preOrderTraversal(f))
        //case _ => throw new Exception(s"Unsupported FIRRTL node ${ast.getClass.getSimpleName}!")
        case _ =>
      }
    }
  }
}
