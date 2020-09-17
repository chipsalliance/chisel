// SPDX-License-Identifier: Apache-2.0

package firrtl.traversals

import firrtl.ir._
import language.implicitConversions

/** Enables FIRRTL IR nodes to use foreach to traverse children IR nodes
  */
object Foreachers {

  /** Statement Foreachers */
  private trait StmtForMagnet {
    def foreach(stmt: Statement): Unit
  }
  private object StmtForMagnet {
    implicit def forStmt(f: Statement => Unit): StmtForMagnet = new StmtForMagnet {
      def foreach(stmt: Statement): Unit = stmt.foreachStmt(f)
    }
    implicit def forExp(f: Expression => Unit): StmtForMagnet = new StmtForMagnet {
      def foreach(stmt: Statement): Unit = stmt.foreachExpr(f)
    }
    implicit def forType(f: Type => Unit): StmtForMagnet = new StmtForMagnet {
      def foreach(stmt: Statement): Unit = stmt.foreachType(f)
    }
    implicit def forString(f: String => Unit): StmtForMagnet = new StmtForMagnet {
      def foreach(stmt: Statement): Unit = stmt.foreachString(f)
    }
    implicit def forInfo(f: Info => Unit): StmtForMagnet = new StmtForMagnet {
      def foreach(stmt: Statement): Unit = stmt.foreachInfo(f)
    }
  }
  implicit class StmtForeach(val _stmt: Statement) extends AnyVal {
    // Using implicit types to allow overloading of function type to foreach, see StmtForMagnet above
    def foreach[T](f: T => Unit)(implicit magnet: (T => Unit) => StmtForMagnet): Unit = magnet(f).foreach(_stmt)
  }

  /** Expression Foreachers */
  private trait ExprForMagnet {
    def foreach(expr: Expression): Unit
  }
  private object ExprForMagnet {
    implicit def forExpr(f: Expression => Unit): ExprForMagnet = new ExprForMagnet {
      def foreach(expr: Expression): Unit = expr.foreachExpr(f)
    }
    implicit def forType(f: Type => Unit): ExprForMagnet = new ExprForMagnet {
      def foreach(expr: Expression): Unit = expr.foreachType(f)
    }
    implicit def forWidth(f: Width => Unit): ExprForMagnet = new ExprForMagnet {
      def foreach(expr: Expression): Unit = expr.foreachWidth(f)
    }
  }
  implicit class ExprForeach(val _expr: Expression) extends AnyVal {
    def foreach[T](f: T => Unit)(implicit magnet: (T => Unit) => ExprForMagnet): Unit = magnet(f).foreach(_expr)
  }

  /** Type Foreachers */
  private trait TypeForMagnet {
    def foreach(tpe: Type): Unit
  }
  private object TypeForMagnet {
    implicit def forType(f: Type => Unit): TypeForMagnet = new TypeForMagnet {
      def foreach(tpe: Type): Unit = tpe.foreachType(f)
    }
    implicit def forWidth(f: Width => Unit): TypeForMagnet = new TypeForMagnet {
      def foreach(tpe: Type): Unit = tpe.foreachWidth(f)
    }
  }
  implicit class TypeForeach(val _tpe: Type) extends AnyVal {
    def foreach[T](f: T => Unit)(implicit magnet: (T => Unit) => TypeForMagnet): Unit = magnet(f).foreach(_tpe)
  }

  /** Module Foreachers */
  private trait ModuleForMagnet {
    def foreach(module: DefModule): Unit
  }
  private object ModuleForMagnet {
    implicit def forStmt(f: Statement => Unit): ModuleForMagnet = new ModuleForMagnet {
      def foreach(module: DefModule): Unit = module.foreachStmt(f)
    }
    implicit def forPorts(f: Port => Unit): ModuleForMagnet = new ModuleForMagnet {
      def foreach(module: DefModule): Unit = module.foreachPort(f)
    }
    implicit def forString(f: String => Unit): ModuleForMagnet = new ModuleForMagnet {
      def foreach(module: DefModule): Unit = module.foreachString(f)
    }
    implicit def forInfo(f: Info => Unit): ModuleForMagnet = new ModuleForMagnet {
      def foreach(module: DefModule): Unit = module.foreachInfo(f)
    }
  }
  implicit class ModuleForeach(val _module: DefModule) extends AnyVal {
    def foreach[T](f: T => Unit)(implicit magnet: (T => Unit) => ModuleForMagnet): Unit = magnet(f).foreach(_module)
  }

  /** Circuit Foreachers */
  private trait CircuitForMagnet {
    def foreach(module: Circuit): Unit
  }
  private object CircuitForMagnet {
    implicit def forModules(f: DefModule => Unit): CircuitForMagnet = new CircuitForMagnet {
      def foreach(circuit: Circuit): Unit = circuit.foreachModule(f)
    }
    implicit def forString(f: String => Unit): CircuitForMagnet = new CircuitForMagnet {
      def foreach(circuit: Circuit): Unit = circuit.foreachString(f)
    }
    implicit def forInfo(f: Info => Unit): CircuitForMagnet = new CircuitForMagnet {
      def foreach(circuit: Circuit): Unit = circuit.foreachInfo(f)
    }
  }
  implicit class CircuitForeach(val _circuit: Circuit) extends AnyVal {
    def foreach[T](f: T => Unit)(implicit magnet: (T => Unit) => CircuitForMagnet): Unit = magnet(f).foreach(_circuit)
  }
}
