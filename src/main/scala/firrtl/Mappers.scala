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

// TODO: Implement remaining mappers and recursive mappers
object Mappers {

  // ********** Stmt Mappers **********
  private trait StmtMagnet {
    def map(stmt: Stmt): Stmt
  }
  private object StmtMagnet {
    implicit def forStmt(f: Stmt => Stmt) = new StmtMagnet {
      override def map(stmt: Stmt): Stmt = {
        stmt match {
          case s: Conditionally => Conditionally(s.info, s.pred, f(s.conseq), f(s.alt))
          case s: Begin => Begin(s.stmts.map(f))
          case s: Stmt => s
        }
      }
    }
    implicit def forExp(f: Expression => Expression) = new StmtMagnet {
      override def map(stmt: Stmt): Stmt = {
        stmt match { 
          case s: DefRegister => DefRegister(s.info, s.name, s.tpe, f(s.clock), f(s.reset), f(s.init))
          case s: DefNode => DefNode(s.info, s.name, f(s.value))
          case s: Connect => Connect(s.info, f(s.loc), f(s.exp))
          case s: BulkConnect => BulkConnect(s.info, f(s.loc), f(s.exp))
          case s: Conditionally => Conditionally(s.info, f(s.pred), s.conseq, s.alt)
          case s: IsInvalid => IsInvalid(s.info, f(s.exp))
          case s: Stop => Stop(s.info, s.ret, f(s.clk), f(s.en))
          case s: Print => Print(s.info, s.string, s.args.map(f), f(s.clk), f(s.en))
          case s: CDefMPort => CDefMPort(s.info,s.name,s.tpe,s.mem,s.exps.map(f),s.direction)
          case s: Stmt => s 
        }
      }
    }
    implicit def forType(f: Type => Type) = new StmtMagnet {
      override def map(stmt: Stmt) : Stmt = {
        stmt match {
          case s:DefPoison => DefPoison(s.info,s.name,f(s.tpe))
          case s:DefWire => DefWire(s.info,s.name,f(s.tpe))
          case s:DefRegister => DefRegister(s.info,s.name,f(s.tpe),s.clock,s.reset,s.init)
          case s:DefMemory => DefMemory(s.info,s.name, f(s.data_type), s.depth, s.write_latency, s.read_latency, s.readers, s.writers, s.readwriters)
          case s:CDefMemory => CDefMemory(s.info,s.name, f(s.tpe), s.size, s.seq)
          case s:CDefMPort => CDefMPort(s.info,s.name, f(s.tpe), s.mem, s.exps,s.direction)
          case s => s
        }
      }
    }
    implicit def forString(f: String => String) = new StmtMagnet {
      override def map(stmt: Stmt): Stmt = {
        stmt match {
          case s: DefWire => DefWire(s.info,f(s.name),s.tpe)
          case s: DefPoison => DefPoison(s.info,f(s.name),s.tpe)
          case s: DefRegister => DefRegister(s.info,f(s.name), s.tpe, s.clock, s.reset, s.init)
          case s: DefMemory => DefMemory(s.info,f(s.name), s.data_type, s.depth, s.write_latency, s.read_latency, s.readers, s.writers, s.readwriters)
          case s: DefNode => DefNode(s.info,f(s.name),s.value)
          case s: DefInstance => DefInstance(s.info,f(s.name), s.module)
          case s: WDefInstance => WDefInstance(s.info,f(s.name), s.module,s.tpe)
          case s: CDefMemory => CDefMemory(s.info,f(s.name),s.tpe,s.size,s.seq)
          case s: CDefMPort => CDefMPort(s.info,f(s.name),s.tpe,s.mem,s.exps,s.direction)
          case s => s
        }
      }
    }
  }
  implicit class StmtMap(stmt: Stmt) {
    // Using implicit types to allow overloading of function type to map, see StmtMagnet above
    def map[T](f: T => T)(implicit magnet: (T => T) => StmtMagnet): Stmt = magnet(f).map(stmt)
  }

  // ********** Expression Mappers **********
  private trait ExpMagnet {
    def map(exp: Expression): Expression
  }
  private object ExpMagnet {
    implicit def forExp(f: Expression => Expression) = new ExpMagnet {
      override def map(exp: Expression): Expression = {
        exp match {
          case e: SubField => SubField(f(e.exp), e.name, e.tpe)
          case e: SubIndex => SubIndex(f(e.exp), e.value, e.tpe)
          case e: SubAccess => SubAccess(f(e.exp), f(e.index), e.tpe)
          case e: Mux => Mux(f(e.cond), f(e.tval), f(e.fval), e.tpe)
          case e: ValidIf => ValidIf(f(e.cond), f(e.value), e.tpe)
          case e: DoPrim => DoPrim(e.op, e.args.map(f), e.consts, e.tpe)
          case e: WSubField => WSubField(f(e.exp), e.name, e.tpe, e.gender)
          case e: WSubIndex => WSubIndex(f(e.exp), e.value, e.tpe, e.gender)
          case e: WSubAccess => WSubAccess(f(e.exp), f(e.index), e.tpe, e.gender)
          case e: Expression => e
        }
      }
    }
    implicit def forType(f: Type => Type) = new ExpMagnet {
      override def map(exp: Expression): Expression = {
        exp match {
          case e: DoPrim => DoPrim(e.op,e.args,e.consts,f(e.tpe))
          case e: Mux => Mux(e.cond,e.tval,e.fval,f(e.tpe))
          case e: ValidIf => ValidIf(e.cond,e.value,f(e.tpe))
          case e: WRef => WRef(e.name,f(e.tpe),e.kind,e.gender)
          case e: WSubField => WSubField(e.exp,e.name,f(e.tpe),e.gender)
          case e: WSubIndex => WSubIndex(e.exp,e.value,f(e.tpe),e.gender)
          case e: WSubAccess => WSubAccess(e.exp,e.index,f(e.tpe),e.gender)
          case e => e
        }
      }
    }
    implicit def forWidth(f: Width => Width) = new ExpMagnet {
      override def map(exp: Expression): Expression = {
        exp match {
          case e: UIntValue => UIntValue(e.value,f(e.width))
          case e: SIntValue => SIntValue(e.value,f(e.width))
          case e => e
        }
      }
    }
  }
  implicit class ExpMap(exp: Expression) {
    def map[T](f: T => T)(implicit magnet: (T => T) => ExpMagnet): Expression = magnet(f).map(exp)
  }

  // ********** Type Mappers **********
  private trait TypeMagnet {
    def map(tpe: Type): Type
  }
  private object TypeMagnet {
    implicit def forType(f: Type => Type) = new TypeMagnet {
      override def map(tpe: Type): Type = {
        tpe match {
          case t: BundleType => BundleType(t.fields.map(p => Field(p.name, p.flip, f(p.tpe))))
          case t: VectorType => VectorType(f(t.tpe), t.size)
          case t => t
        }
      }
    }
    implicit def forWidth(f: Width => Width) = new TypeMagnet {
      override def map(tpe: Type): Type = {
        tpe match {
          case t: UIntType => UIntType(f(t.width))
          case t: SIntType => SIntType(f(t.width))
          case t => t
        }
      }
    }
  }
  implicit class TypeMap(tpe: Type) {
    def map[T](f: T => T)(implicit magnet: (T => T) => TypeMagnet): Type = magnet(f).map(tpe)
  }

  // ********** Width Mappers **********
  private trait WidthMagnet {
    def map(width: Width): Width
  }
  private object WidthMagnet {
    implicit def forWidth(f: Width => Width) = new WidthMagnet {
      override def map(width: Width): Width = {
        width match {
          case w: MaxWidth => MaxWidth(w.args.map(f))
          case w: MinWidth => MinWidth(w.args.map(f))
          case w: PlusWidth => PlusWidth(f(w.arg1),f(w.arg2))
          case w: MinusWidth => MinusWidth(f(w.arg1),f(w.arg2))
          case w: ExpWidth => ExpWidth(f(w.arg1))
          case w => w
        }
      }
    }
  }
  implicit class WidthMap(width: Width) {
    def map[T](f: T => T)(implicit magnet: (T => T) => WidthMagnet): Width = magnet(f).map(width)
  }

}
