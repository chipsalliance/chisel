// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.internal.firrtl.Width
import chisel3.testers.BasicTester

import chisel3.experimental._

//trait BundleEltsSpecUtils {
//  class BundleWidthModule extends Module {
//    class BundleWidth[T <: Data](val genType: T, widthx: Width) extends Bundle {
//      val gen = genType.chiselCloneType
//      val in = Input(UInt(widthx))
//      val out = Output(UInt(widthx))
//
//      def getIntWidth = widthx
////      override def cloneType = new BundleWidth(genType, width).asInstanceOf[this.type]
//
//      def GPF(rootClass: Class[_]) = {
//        import reflect.runtime._,universe._
//        val im = currentMirror reflect this
//        val vals = im.symbol.asClass.typeSignature.members filter (s => s.isTerm && s.asTerm.isAccessor)
//        val methods = vals map (im reflectMethod _.asMethod)
//
//        // Suggest names to nodes using runtime reflection
//        def getValFields(c: Class[_]): Set[java.lang.reflect.Field] = {
//          if (c == rootClass) Set()
//          else getValFields(c.getSuperclass) ++ c.getDeclaredFields
//        }
//        val valNames = getValFields(this.getClass).map(_.getName)
//        def isPublicVal(m: java.lang.reflect.Method) =
//          m.getParameterTypes.isEmpty && valNames.contains(m.getName) && m.getDeclaringClass.isAssignableFrom(rootClass)
//        getValFields(this.getClass).map(x => print(s"VN ${x.getName} => ${x.get(this)}\r\n"))
//        this.getClass.getMethods.map(x => print(s"GM ${x.getName}\r\n"))
//        this.getClass.getMethods.sortWith(_.getName < _.getName).filter(isPublicVal(_))
//      }
//    }
//    val io = IO(new BundleWidth(UInt(2.W), 2.W))
////    val int = io.chiselCloneType
//    io.out := io.in + 1.U
//
//    import chisel3.core.DataMirror
//    print(s"GenWidth ${DataMirror.widthOf(io)}\r\n")
//    print(s"GPF ${io.GPF(classOf[Bundle])}\r\n")
//  }
//
//  @chiselName
//  @dump
//  class MySimpleModule extends Module {
//    val io = IO(new Bundle {
//      val in = Input(UInt(32.W))
//      val out = Output(UInt(32.W))
//    })
//
//    for (i <- 0 until 1) {
//      val a = Wire(init = io.in)
//      val b = Wire(init = a)
//      io.out := b
//    }
//  }
//
//  @chiselName
//  @dump
//  class MySimpleAFodule extends Module {
//    val io = IO(new Bundle {
//      val in = Input(UInt(32.W))
//      val out = Output(UInt(32.W))
//    })
//
//    Seq(io.in).foreach { in =>
//      val a = Wire(init = in)
//      val b = Wire(init = a)
//      io.out := b
//    }
//  }
//}
//
//class BundleEltsSpec extends ChiselFlatSpec with BundleEltsSpecUtils {
//  "Bundles with the same fields but in different orders" should "bulk connect" in {
//    elaborate { new BundleWidthModule }
//  }
//}
