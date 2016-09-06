/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

package firrtl

import firrtl.ir._

// TODO: Implement remaining mappers and recursive mappers
object Mappers {

  // ********** Stmt Mappers **********
  private trait StmtMagnet {
    def map(stmt: Statement): Statement
  }
  private object StmtMagnet {
    implicit def forStmt(f: Statement => Statement) = new StmtMagnet {
      override def map(stmt: Statement): Statement = stmt mapStmt f
    }
    implicit def forExp(f: Expression => Expression) = new StmtMagnet {
      override def map(stmt: Statement): Statement = stmt mapExpr f
    }
    implicit def forType(f: Type => Type) = new StmtMagnet {
      override def map(stmt: Statement) : Statement = stmt mapType f
    }
    implicit def forString(f: String => String) = new StmtMagnet {
      override def map(stmt: Statement): Statement = stmt mapString f
    }
  }
  implicit class StmtMap(val stmt: Statement) extends AnyVal {
    // Using implicit types to allow overloading of function type to map, see StmtMagnet above
    def map[T](f: T => T)(implicit magnet: (T => T) => StmtMagnet): Statement = magnet(f).map(stmt)
  }

  // ********** Expression Mappers **********
  private trait ExprMagnet {
    def map(expr: Expression): Expression
  }
  private object ExprMagnet {
    implicit def forExpr(f: Expression => Expression) = new ExprMagnet {
      override def map(expr: Expression): Expression = expr mapExpr f
    }
    implicit def forType(f: Type => Type) = new ExprMagnet {
      override def map(expr: Expression): Expression = expr mapType f
    }
    implicit def forWidth(f: Width => Width) = new ExprMagnet {
      override def map(expr: Expression): Expression = expr mapWidth f
    }
  }
  implicit class ExprMap(val expr: Expression) extends AnyVal {
    def map[T](f: T => T)(implicit magnet: (T => T) => ExprMagnet): Expression = magnet(f).map(expr)
  }

  // ********** Type Mappers **********
  private trait TypeMagnet {
    def map(tpe: Type): Type
  }
  private object TypeMagnet {
    implicit def forType(f: Type => Type) = new TypeMagnet {
      override def map(tpe: Type): Type = tpe mapType f
    }
    implicit def forWidth(f: Width => Width) = new TypeMagnet {
      override def map(tpe: Type): Type = tpe mapWidth f
    }
  }
  implicit class TypeMap(val tpe: Type) extends AnyVal {
    def map[T](f: T => T)(implicit magnet: (T => T) => TypeMagnet): Type = magnet(f).map(tpe)
  }

  // ********** Width Mappers **********
  private trait WidthMagnet {
    def map(width: Width): Width
  }
  private object WidthMagnet {
    implicit def forWidth(f: Width => Width) = new WidthMagnet {
      override def map(width: Width): Width = width match {
        case mapable: HasMapWidth => mapable mapWidth f // WIR
        case other => other // Standard IR nodes
      }
    }
  }
  implicit class WidthMap(val width: Width) extends AnyVal {
    def map[T](f: T => T)(implicit magnet: (T => T) => WidthMagnet): Width = magnet(f).map(width)
  }

  // ********** Module Mappers **********
  private trait ModuleMagnet {
    def map(module: DefModule): DefModule
  }
  private object ModuleMagnet {
    implicit def forStmt(f: Statement => Statement) = new ModuleMagnet {
      override def map(module: DefModule): DefModule = module mapStmt f
    }
    implicit def forPorts(f: Port => Port) = new ModuleMagnet {
      override def map(module: DefModule): DefModule = module mapPort f
    }
    implicit def forString(f: String => String) = new ModuleMagnet {
      override def map(module: DefModule): DefModule = module mapString f
    }
  }
  implicit class ModuleMap(val module: DefModule) extends AnyVal {
    def map[T](f: T => T)(implicit magnet: (T => T) => ModuleMagnet): DefModule = magnet(f).map(module)
  }

}
