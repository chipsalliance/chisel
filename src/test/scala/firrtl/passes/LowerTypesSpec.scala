// SPDX-License-Identifier: Apache-2.0

package firrtl.passes
import firrtl.annotations.{CircuitTarget, IsMember}
import firrtl.{CircuitState, RenameMap, Utils}
import firrtl.options.Dependency
import firrtl.stage.TransformManager
import firrtl.stage.TransformManager.TransformDependency
import org.scalatest.flatspec.AnyFlatSpec

/** Unit test style tests for [[LowerTypes]].
  * You can find additional integration style tests in [[firrtlTests.LowerTypesSpec]]
  */
class LowerTypesUnitTestSpec extends LowerTypesBaseSpec {
  import LowerTypesSpecUtils._
  override protected def lower(n: String, tpe: String, namespace: Set[String]): Seq[String] =
    destruct(n, tpe, namespace).fields
}

/** Runs the lowering pass in the context of the compiler instead of directly calling internal functions. */
class LowerTypesEndToEndSpec extends LowerTypesBaseSpec {
  private lazy val lowerTypesCompiler = new TransformManager(Seq(Dependency(LowerTypes)))
  private def legacyLower(n: String, tpe: String, namespace: Set[String]): Seq[String] = {
    val inputs = namespace.map(n => s"    input $n : UInt<1>").mkString("\n")
    val src =
      s"""circuit c:
         |  module c:
         |$inputs
         |    output $n : $tpe
         |    $n is invalid
         |""".stripMargin
    val c = CircuitState(firrtl.Parser.parse(src), Seq())
    val c2 = lowerTypesCompiler.execute(c)
    val ps = c2.circuit.modules.head.ports.filterNot(p => namespace.contains(p.name))
    ps.map { p =>
      val orientation = Utils.to_flip(p.direction)
      s"${orientation.serialize}${p.name} : ${p.tpe.serialize}"
    }
  }

  override protected def lower(n: String, tpe: String, namespace: Set[String]): Seq[String] =
    legacyLower(n, tpe, namespace)
}

/** this spec can be tested with either the new or the old LowerTypes pass */
abstract class LowerTypesBaseSpec extends AnyFlatSpec {
  protected def lower(n: String, tpe: String, namespace: Set[String] = Set()): Seq[String]

  it should "lower bundles and vectors" in {
    assert(lower("a", "{ a : UInt<1>, b : UInt<1>}") == Seq("a_a : UInt<1>", "a_b : UInt<1>"))
    assert(lower("a", "{ a : UInt<1>, b : { c : UInt<1>}}") == Seq("a_a : UInt<1>", "a_b_c : UInt<1>"))
    assert(lower("a", "{ a : UInt<1>, b : UInt<1>[2]}") == Seq("a_a : UInt<1>", "a_b_0 : UInt<1>", "a_b_1 : UInt<1>"))
    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>}[2]") ==
        Seq("a_0_a : UInt<1>", "a_0_b : UInt<1>", "a_1_a : UInt<1>", "a_1_b : UInt<1>")
    )

    // with conflicts
    assert(lower("a", "{ a : UInt<1>, b : UInt<1>}", Set("a_a")) == Seq("a__a : UInt<1>", "a__b : UInt<1>"))
    assert(lower("a", "{ a : UInt<1>, b : UInt<1>}", Set("a_b")) == Seq("a__a : UInt<1>", "a__b : UInt<1>"))
    assert(lower("a", "{ a : UInt<1>, b : UInt<1>}", Set("a_c")) == Seq("a_a : UInt<1>", "a_b : UInt<1>"))

    assert(lower("a", "{ a : UInt<1>, b : { c : UInt<1>}}", Set("a_a")) == Seq("a__a : UInt<1>", "a__b_c : UInt<1>"))
    // in this case we do not have a "real" conflict, but it could be in a reference and thus a is still changed to a_
    assert(lower("a", "{ a : UInt<1>, b : { c : UInt<1>}}", Set("a_b")) == Seq("a__a : UInt<1>", "a__b_c : UInt<1>"))
    assert(lower("a", "{ a : UInt<1>, b : { c : UInt<1>}}", Set("a_b_c")) == Seq("a__a : UInt<1>", "a__b_c : UInt<1>"))

    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>[2]}", Set("a_a")) ==
        Seq("a__a : UInt<1>", "a__b_0 : UInt<1>", "a__b_1 : UInt<1>")
    )
    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>[2]}", Set("a_a", "a_b_0")) ==
        Seq("a__a : UInt<1>", "a__b_0 : UInt<1>", "a__b_1 : UInt<1>")
    )
    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>[2]}", Set("a_b_0")) ==
        Seq("a__a : UInt<1>", "a__b_0 : UInt<1>", "a__b_1 : UInt<1>")
    )

    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>}[2]", Set("a_0")) ==
        Seq("a__0_a : UInt<1>", "a__0_b : UInt<1>", "a__1_a : UInt<1>", "a__1_b : UInt<1>")
    )
    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>}[2]", Set("a_3")) ==
        Seq("a_0_a : UInt<1>", "a_0_b : UInt<1>", "a_1_a : UInt<1>", "a_1_b : UInt<1>")
    )
    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>}[2]", Set("a_0_a")) ==
        Seq("a__0_a : UInt<1>", "a__0_b : UInt<1>", "a__1_a : UInt<1>", "a__1_b : UInt<1>")
    )
    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>}[2]", Set("a_0_c")) ==
        Seq("a_0_a : UInt<1>", "a_0_b : UInt<1>", "a_1_a : UInt<1>", "a_1_b : UInt<1>")
    )

    // collisions inside the bundle
    assert(
      lower("a", "{ a : UInt<1>, b : { c : UInt<1>}, b_c : UInt<1>}") ==
        Seq("a_a : UInt<1>", "a_b__c : UInt<1>", "a_b_c : UInt<1>")
    )
    assert(
      lower("a", "{ a : UInt<1>, b : { c : UInt<1>}, b_b : UInt<1>}") ==
        Seq("a_a : UInt<1>", "a_b_c : UInt<1>", "a_b_b : UInt<1>")
    )

    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>[2], b_0 : UInt<1>}") ==
        Seq("a_a : UInt<1>", "a_b__0 : UInt<1>", "a_b__1 : UInt<1>", "a_b_0 : UInt<1>")
    )
    assert(
      lower("a", "{ a : UInt<1>, b : UInt<1>[2], b_c : UInt<1>}") ==
        Seq("a_a : UInt<1>", "a_b_0 : UInt<1>", "a_b_1 : UInt<1>", "a_b_c : UInt<1>")
    )
  }

  it should "correctly lower the orientation" in {
    assert(lower("a", "{ flip a : UInt<1>, b : UInt<1>}") == Seq("flip a_a : UInt<1>", "a_b : UInt<1>"))
    assert(
      lower("a", "{ flip a : UInt<1>[2], b : UInt<1>}") ==
        Seq("flip a_a_0 : UInt<1>", "flip a_a_1 : UInt<1>", "a_b : UInt<1>")
    )
    assert(
      lower("a", "{ a : { flip c : UInt<1>, d : UInt<1>}[2], b : UInt<1>}") ==
        Seq(
          "flip a_a_0_c : UInt<1>",
          "a_a_0_d : UInt<1>",
          "flip a_a_1_c : UInt<1>",
          "a_a_1_d : UInt<1>",
          "a_b : UInt<1>"
        )
    )
  }
}

/** Test the renaming for "regular" references, i.e. Wires, Nodes and Register.
  * Memories and Instances are special cases.
  */
class LowerTypesRenamingSpec extends AnyFlatSpec {
  import LowerTypesSpecUtils._
  protected def lower(n: String, tpe: String, namespace: Set[String] = Set()): RenameMap =
    destruct(n, tpe, namespace).renameMap

  private val m = CircuitTarget("m").module("m")

  it should "not rename ground types" in {
    val r = lower("a", "UInt<1>")
    assert(r.underlying.isEmpty)
  }

  it should "properly rename lowered bundles and vectors" in {
    val a = m.ref("a")

    def one(namespace: Set[String], prefix: String): Unit = {
      val r = lower("a", "{ a : UInt<1>, b : UInt<1>}", namespace)
      assert(get(r, a) == Set(m.ref(prefix + "a"), m.ref(prefix + "b")))
      assert(get(r, a.field("a")) == Set(m.ref(prefix + "a")))
      assert(get(r, a.field("b")) == Set(m.ref(prefix + "b")))
    }
    one(Set(), "a_")
    one(Set("a_a"), "a__")

    def two(namespace: Set[String], prefix: String): Unit = {
      val r = lower("a", "{ a : UInt<1>, b : { c : UInt<1>}}", namespace)
      assert(get(r, a) == Set(m.ref(prefix + "a"), m.ref(prefix + "b_c")))
      assert(get(r, a.field("a")) == Set(m.ref(prefix + "a")))
      assert(get(r, a.field("b")) == Set(m.ref(prefix + "b_c")))
      assert(get(r, a.field("b").field("c")) == Set(m.ref(prefix + "b_c")))
    }
    two(Set(), "a_")
    two(Set("a_a"), "a__")

    def three(namespace: Set[String], prefix: String): Unit = {
      val r = lower("a", "{ a : UInt<1>, b : UInt<1>[2]}", namespace)
      assert(get(r, a) == Set(m.ref(prefix + "a"), m.ref(prefix + "b_0"), m.ref(prefix + "b_1")))
      assert(get(r, a.field("a")) == Set(m.ref(prefix + "a")))
      assert(get(r, a.field("b")) == Set(m.ref(prefix + "b_0"), m.ref(prefix + "b_1")))
      assert(get(r, a.field("b").index(0)) == Set(m.ref(prefix + "b_0")))
      assert(get(r, a.field("b").index(1)) == Set(m.ref(prefix + "b_1")))
    }
    three(Set(), "a_")
    three(Set("a_b_0"), "a__")

    def four(namespace: Set[String], prefix: String): Unit = {
      val r = lower("a", "{ a : UInt<1>, b : UInt<1>}[2]", namespace)
      assert(
        get(r, a) == Set(m.ref(prefix + "0_a"), m.ref(prefix + "1_a"), m.ref(prefix + "0_b"), m.ref(prefix + "1_b"))
      )
      assert(get(r, a.index(0)) == Set(m.ref(prefix + "0_a"), m.ref(prefix + "0_b")))
      assert(get(r, a.index(1)) == Set(m.ref(prefix + "1_a"), m.ref(prefix + "1_b")))
      assert(get(r, a.index(0).field("a")) == Set(m.ref(prefix + "0_a")))
      assert(get(r, a.index(0).field("b")) == Set(m.ref(prefix + "0_b")))
      assert(get(r, a.index(1).field("a")) == Set(m.ref(prefix + "1_a")))
      assert(get(r, a.index(1).field("b")) == Set(m.ref(prefix + "1_b")))
    }
    four(Set(), "a_")
    four(Set("a_0"), "a__")
    four(Set("a_3"), "a_")

    // collisions inside the bundle
    {
      val r = lower("a", "{ a : UInt<1>, b : { c : UInt<1>}, b_c : UInt<1>}")
      assert(get(r, a) == Set(m.ref("a_a"), m.ref("a_b__c"), m.ref("a_b_c")))
      assert(get(r, a.field("a")) == Set(m.ref("a_a")))
      assert(get(r, a.field("b")) == Set(m.ref("a_b__c")))
      assert(get(r, a.field("b").field("c")) == Set(m.ref("a_b__c")))
      assert(get(r, a.field("b_c")) == Set(m.ref("a_b_c")))
    }
    {
      val r = lower("a", "{ a : UInt<1>, b : { c : UInt<1>}, b_b : UInt<1>}")
      assert(get(r, a) == Set(m.ref("a_a"), m.ref("a_b_c"), m.ref("a_b_b")))
      assert(get(r, a.field("a")) == Set(m.ref("a_a")))
      assert(get(r, a.field("b")) == Set(m.ref("a_b_c")))
      assert(get(r, a.field("b").field("c")) == Set(m.ref("a_b_c")))
      assert(get(r, a.field("b_b")) == Set(m.ref("a_b_b")))
    }
    {
      val r = lower("a", "{ a : UInt<1>, b : UInt<1>[2], b_0 : UInt<1>}")
      assert(get(r, a) == Set(m.ref("a_a"), m.ref("a_b__0"), m.ref("a_b__1"), m.ref("a_b_0")))
      assert(get(r, a.field("a")) == Set(m.ref("a_a")))
      assert(get(r, a.field("b")) == Set(m.ref("a_b__0"), m.ref("a_b__1")))
      assert(get(r, a.field("b").index(0)) == Set(m.ref("a_b__0")))
      assert(get(r, a.field("b").index(1)) == Set(m.ref("a_b__1")))
      assert(get(r, a.field("b_0")) == Set(m.ref("a_b_0")))
    }
  }
}

/** Instances are a special case since they do not get completely destructed but instead become a 1-deep bundle. */
class LowerTypesOfInstancesSpec extends AnyFlatSpec {
  import LowerTypesSpecUtils._
  private case class Lower(inst: firrtl.ir.DefInstance, fields: Seq[String], renameMap: RenameMap)
  private val m = CircuitTarget("m").module("m")
  def resultToFieldSeq(res: Seq[(String, firrtl.ir.SubField)]): Seq[String] =
    res.map(_._2).map(r => s"${r.name} : ${r.tpe.serialize}")
  private def lower(
    n:         String,
    tpe:       String,
    module:    String,
    namespace: Set[String],
    renames:   RenameMap = RenameMap()
  ): Lower = {
    val ref = firrtl.ir.DefInstance(firrtl.ir.NoInfo, n, module, parseType(tpe))
    val mutableSet = scala.collection.mutable.HashSet[String]() ++ namespace
    val (newInstance, res) = DestructTypes.destructInstance(m, ref, mutableSet, renames, Set())
    Lower(newInstance, resultToFieldSeq(res), renames)
  }
  private def get(l: Lower, m: IsMember): Set[IsMember] = l.renameMap.get(m).get.toSet

  it should "not rename instances if the instance name does not change" in {
    val l = lower("i", "{ a : UInt<1>}", "c", Set())
    assert(l.renameMap.underlying.isEmpty)
  }

  it should "lower an instance correctly" in {
    val i = m.instOf("i", "c")
    val l = lower("i", "{ a : UInt<1>}", "c", Set("i_a"))
    assert(l.inst.name == "i_")
    assert(l.inst.tpe.isInstanceOf[firrtl.ir.BundleType])
    assert(l.inst.tpe.serialize == "{ a : UInt<1>}")

    assert(get(l, i) == Set(m.instOf("i_", "c")))
    assert(l.fields == Seq("a : UInt<1>"))
  }

  it should "update the rename map with the changed port names" in {
    // without lowering ports
    {
      val i = m.instOf("i", "c")
      val l = lower("i", "{ b : { c : UInt<1>}, b_c : UInt<1>}", "c", Set("i_b_c"))
      // the instance was renamed because of the collision with "i_b_c"
      assert(get(l, i) == Set(m.instOf("i_", "c")))
      // the rename of e.g. `instance.b` to `instance_.b__c` was not recorded since we never performed the
      // port renaming and thus we won't get a result
      assert(get(l, i.ref("b")) == Set(m.instOf("i_", "c").ref("b")))
    }

    // same as above but with lowered port
    {
      // We need two distinct rename maps: one for the port renaming and one for everything else.
      // This is to accommodate the use-case where a port as well as an instance needs to be renames
      // thus requiring a two-stage translation process for reference to the port of the instance.
      // This two-stage translation is only supported through chaining rename maps.
      val portRenames = RenameMap()
      val otherRenames = RenameMap()

      // The child module "c" which we assume has the following ports: b : { c : UInt<1>} and b_c : UInt<1>
      val c = CircuitTarget("m").module("c")
      val portB = firrtl.ir.Field("b", firrtl.ir.Default, parseType("{ c : UInt<1>}"))
      val portB_C = firrtl.ir.Field("b_c", firrtl.ir.Default, parseType("UInt<1>"))

      // lower ports
      val namespaceC = scala.collection.mutable.HashSet[String]() ++ Seq("b", "b_c")
      DestructTypes.destruct(c, portB, namespaceC, portRenames, Set())
      DestructTypes.destruct(c, portB_C, namespaceC, portRenames, Set())
      // only port b is renamed, port b_c stays the same
      assert(portRenames.get(c.ref("b")).get == Seq(c.ref("b__c")))

      // in module m we then lower the instance i of c
      val l = lower("i", "{ b : { c : UInt<1>}, b_c : UInt<1>}", "c", Set("i_b_c"), otherRenames)
      val i = m.instOf("i", "c")
      // the instance was renamed because of the collision with "i_b_c"
      val i_ = m.instOf("i_", "c")
      assert(get(l, i) == Set(i_))

      // the ports renaming is also noted
      val r = portRenames.andThen(otherRenames)
      assert(r.get(i.ref("b")).get == Seq(i_.ref("b__c")))
      assert(r.get(i.ref("b").field("c")).get == Seq(i_.ref("b__c")))
      assert(r.get(i.ref("b_c")).get == Seq(i_.ref("b_c")))
    }
  }
}

/** Memories are a special case as they remain 2-deep bundles and fields of the datatype are pulled into the front.
  * E.g., `mem.r.data.a` becomes `mem_a.r.data`
  */
class LowerTypesOfMemorySpec extends AnyFlatSpec {
  import LowerTypesSpecUtils._
  private case class Lower(
    mems:      Seq[firrtl.ir.DefMemory],
    refs:      Seq[(String, firrtl.ir.SubField)],
    renameMap: RenameMap)
  private val m = CircuitTarget("m").module("m")
  private val mem = m.ref("mem")
  private def lower(
    name:      String,
    tpe:       String,
    namespace: Set[String],
    r:         Seq[String] = List("r"),
    w:         Seq[String] = List("w"),
    rw:        Seq[String] = List(),
    depth:     Int = 2
  ): Lower = {
    val dataType = parseType(tpe)
    val mem = firrtl.ir.DefMemory(
      firrtl.ir.NoInfo,
      name,
      dataType,
      depth = depth,
      writeLatency = 1,
      readLatency = 1,
      readUnderWrite = firrtl.ir.ReadUnderWrite.Undefined,
      readers = r,
      writers = w,
      readwriters = rw
    )
    val renames = RenameMap()
    val mutableSet = scala.collection.mutable.HashSet[String]() ++ namespace
    val (mems, refs) = DestructTypes.destructMemory(m, mem, mutableSet, renames, Set())
    Lower(mems, refs, renames)
  }
  private val UInt1 = firrtl.ir.UIntType(firrtl.ir.IntWidth(1))

  it should "not rename anything for a ground type memory if there was no conflict" in {
    val l = lower("mem", "UInt<1>", Set("mem_r", "mem_r_data"), w = Seq("w"))
    assert(l.renameMap.underlying.isEmpty)
  }

  it should "still produce reference lookups, even for a ground type memory with no conflicts" in {
    val nameToRef = lower("mem", "UInt<1>", Set("mem_r", "mem_r_data"), w = Seq("w")).refs.map {
      case (n, r) => n -> r.serialize
    }.toSet

    assert(
      nameToRef == Set(
        "mem.r.clk" -> "mem.r.clk",
        "mem.r.en" -> "mem.r.en",
        "mem.r.addr" -> "mem.r.addr",
        "mem.r.data" -> "mem.r.data",
        "mem.w.clk" -> "mem.w.clk",
        "mem.w.en" -> "mem.w.en",
        "mem.w.addr" -> "mem.w.addr",
        "mem.w.data" -> "mem.w.data",
        "mem.w.mask" -> "mem.w.mask"
      )
    )
  }

  it should "produce references of correct type" in {
    val nameToType = lower("mem", "UInt<4>", Set("mem_r", "mem_r_data"), w = Seq("w"), depth = 3).refs.map {
      case (n, r) => n -> r.tpe.serialize
    }.toSet

    assert(
      nameToType == Set(
        "mem.r.clk" -> "Clock",
        "mem.r.en" -> "UInt<1>",
        "mem.r.addr" -> "UInt<2>", // depth = 3
        "mem.r.data" -> "UInt<4>",
        "mem.w.clk" -> "Clock",
        "mem.w.en" -> "UInt<1>",
        "mem.w.addr" -> "UInt<2>",
        "mem.w.data" -> "UInt<4>",
        "mem.w.mask" -> "UInt<1>"
      )
    )
  }

  it should "not rename ground type memories even if there are conflicts on the ports" in {
    // There actually isn't such a thing as conflicting ports, because they do not get flattened by LowerTypes.
    val r = lower("mem", "UInt<1>", Set("mem_r", "mem_r_data"), w = Seq("r_data")).renameMap
    assert(r.underlying.isEmpty)
  }

  it should "rename references to lowered ports" in {
    val r = lower("mem", "{ a : UInt<1>, b : UInt<1>}", Set("mem_a"), r = Seq("r", "r_data")).renameMap

    // complete memory
    assert(get(r, mem) == Set(m.ref("mem__a"), m.ref("mem__b")))

    // read ports
    assert(
      get(r, mem.field("r")) ==
        Set(m.ref("mem__a").field("r"), m.ref("mem__b").field("r"))
    )
    assert(
      get(r, mem.field("r_data")) ==
        Set(m.ref("mem__a").field("r_data"), m.ref("mem__b").field("r_data"))
    )

    // port fields
    assert(
      get(r, mem.field("r").field("data")) ==
        Set(m.ref("mem__a").field("r").field("data"), m.ref("mem__b").field("r").field("data"))
    )
    assert(
      get(r, mem.field("r").field("addr")) ==
        Set(m.ref("mem__a").field("r").field("addr"), m.ref("mem__b").field("r").field("addr"))
    )
    assert(
      get(r, mem.field("r").field("en")) ==
        Set(m.ref("mem__a").field("r").field("en"), m.ref("mem__b").field("r").field("en"))
    )
    assert(
      get(r, mem.field("r").field("clk")) ==
        Set(m.ref("mem__a").field("r").field("clk"), m.ref("mem__b").field("r").field("clk"))
    )
    assert(
      get(r, mem.field("w").field("mask")) ==
        Set(m.ref("mem__a").field("w").field("mask"), m.ref("mem__b").field("w").field("mask"))
    )

    // port sub-fields
    assert(
      get(r, mem.field("r").field("data").field("a")) ==
        Set(m.ref("mem__a").field("r").field("data"))
    )
    assert(
      get(r, mem.field("r").field("data").field("b")) ==
        Set(m.ref("mem__b").field("r").field("data"))
    )

    // need to rename the following:
    // mem -> mem__a, mem__b
    // mem.r.data.{a,b} -> mem__{a,b}.r.data
    // mem.w.data.{a,b} -> mem__{a,b}.w.data
    // mem.w.mask.{a,b} -> mem__{a,b}.w.mask
    // mem.r_data.data.{a,b} -> mem__{a,b}.r_data.data
    val renameCount = r.underlying.map(_._2.size).sum
    assert(renameCount == 10, "it is enough to rename *to* 10 different signals")
    assert(r.underlying.size == 9, "it is enough to rename (from) 9 different signals")
  }

  it should "rename references for a memory with a nested data type" in {
    val l = lower("mem", "{ a : UInt<1>, b : { c : UInt<1>} }", Set("mem_a"))
    assert(l.mems.map(_.name) == Seq("mem__a", "mem__b_c"))
    assert(l.mems.map(_.dataType) == Seq(UInt1, UInt1))

    // complete memory
    val r = l.renameMap
    assert(get(r, mem) == Set(m.ref("mem__a"), m.ref("mem__b_c")))

    // read port
    assert(
      get(r, mem.field("r")) ==
        Set(m.ref("mem__a").field("r"), m.ref("mem__b_c").field("r"))
    )

    // port sub-fields
    assert(
      get(r, mem.field("r").field("data").field("a")) ==
        Set(m.ref("mem__a").field("r").field("data"))
    )
    assert(
      get(r, mem.field("r").field("data").field("b")) ==
        Set(m.ref("mem__b_c").field("r").field("data"))
    )
    assert(
      get(r, mem.field("r").field("data").field("b").field("c")) ==
        Set(m.ref("mem__b_c").field("r").field("data"))
    )

    // the mask field needs to be lowered just like the data field
    assert(
      get(r, mem.field("w").field("mask").field("a")) ==
        Set(m.ref("mem__a").field("w").field("mask"))
    )
    assert(
      get(r, mem.field("w").field("mask").field("b")) ==
        Set(m.ref("mem__b_c").field("w").field("mask"))
    )
    assert(
      get(r, mem.field("w").field("mask").field("b").field("c")) ==
        Set(m.ref("mem__b_c").field("w").field("mask"))
    )

    val renameCount = r.underlying.map(_._2.size).sum
    assert(renameCount == 11, "it is enough to rename *to* 11 different signals")
    assert(r.underlying.size == 10, "it is enough to rename (from) 10 different signals")
  }

  it should "return a name to RefLikeExpression map for a memory with a nested data type" in {
    val nameToRef = lower("mem", "{ a : UInt<1>, b : { c : UInt<1>} }", Set("mem_a")).refs.map {
      case (n, r) => n -> r.serialize
    }.toSet

    assert(
      nameToRef == Set(
        // The non "data" or "mask" fields of read and write ports are already of ground type but still do get duplicated.
        // They will all carry the exact same value, so for a RHS use of the old signal, any of the expanded ones will do.
        "mem.r.clk" -> "mem__a.r.clk",
        "mem.r.clk" -> "mem__b_c.r.clk",
        "mem.r.en" -> "mem__a.r.en",
        "mem.r.en" -> "mem__b_c.r.en",
        "mem.r.addr" -> "mem__a.r.addr",
        "mem.r.addr" -> "mem__b_c.r.addr",
        "mem.w.clk" -> "mem__a.w.clk",
        "mem.w.clk" -> "mem__b_c.w.clk",
        "mem.w.en" -> "mem__a.w.en",
        "mem.w.en" -> "mem__b_c.w.en",
        "mem.w.addr" -> "mem__a.w.addr",
        "mem.w.addr" -> "mem__b_c.w.addr",
        // Ground type references to the data or mask field are unique.
        "mem.r.data.a" -> "mem__a.r.data",
        "mem.w.data.a" -> "mem__a.w.data",
        "mem.w.mask.a" -> "mem__a.w.mask",
        "mem.r.data.b.c" -> "mem__b_c.r.data",
        "mem.w.data.b.c" -> "mem__b_c.w.data",
        "mem.w.mask.b.c" -> "mem__b_c.w.mask"
      )
    )
  }

  it should "produce references of correct type for memories with a read/write port" in {
    val refs = lower(
      "mem",
      "{ a : UInt<3>, b : { c : UInt<4>} }",
      Set("mem_a"),
      r = Seq(),
      w = Seq(),
      rw = Seq("rw"),
      depth = 3
    ).refs
    val nameToRef = refs.map { case (n, r) => n -> r.serialize }.toSet
    val nameToType = refs.map { case (n, r) => n -> r.tpe.serialize }.toSet

    assert(
      nameToRef == Set(
        // The non "data" or "mask" fields of read and write ports are already of ground type but still do get duplicated.
        // They will all carry the exact same value, so for a RHS use of the old signal, any of the expanded ones will do.
        "mem.rw.clk" -> "mem__a.rw.clk",
        "mem.rw.clk" -> "mem__b_c.rw.clk",
        "mem.rw.en" -> "mem__a.rw.en",
        "mem.rw.en" -> "mem__b_c.rw.en",
        "mem.rw.addr" -> "mem__a.rw.addr",
        "mem.rw.addr" -> "mem__b_c.rw.addr",
        "mem.rw.wmode" -> "mem__a.rw.wmode",
        "mem.rw.wmode" -> "mem__b_c.rw.wmode",
        // Ground type references to the data or mask field are unique.
        "mem.rw.rdata.a" -> "mem__a.rw.rdata",
        "mem.rw.wdata.a" -> "mem__a.rw.wdata",
        "mem.rw.wmask.a" -> "mem__a.rw.wmask",
        "mem.rw.rdata.b.c" -> "mem__b_c.rw.rdata",
        "mem.rw.wdata.b.c" -> "mem__b_c.rw.wdata",
        "mem.rw.wmask.b.c" -> "mem__b_c.rw.wmask"
      )
    )

    assert(
      nameToType == Set(
        //
        "mem.rw.clk" -> "Clock",
        "mem.rw.en" -> "UInt<1>",
        "mem.rw.addr" -> "UInt<2>",
        "mem.rw.wmode" -> "UInt<1>",
        // Ground type references to the data or mask field are unique.
        "mem.rw.rdata.a" -> "UInt<3>",
        "mem.rw.wdata.a" -> "UInt<3>",
        "mem.rw.wmask.a" -> "UInt<1>",
        "mem.rw.rdata.b.c" -> "UInt<4>",
        "mem.rw.wdata.b.c" -> "UInt<4>",
        "mem.rw.wmask.b.c" -> "UInt<1>"
      )
    )
  }

  it should "rename references for vector type memories" in {
    val l = lower("mem", "UInt<1>[2]", Set("mem_0"))
    assert(l.mems.map(_.name) == Seq("mem__0", "mem__1"))
    assert(l.mems.map(_.dataType) == Seq(UInt1, UInt1))

    // complete memory
    val r = l.renameMap
    assert(get(r, mem) == Set(m.ref("mem__0"), m.ref("mem__1")))

    // read port
    assert(
      get(r, mem.field("r")) ==
        Set(m.ref("mem__0").field("r"), m.ref("mem__1").field("r"))
    )

    // port sub-fields
    assert(
      get(r, mem.field("r").field("data").index(0)) ==
        Set(m.ref("mem__0").field("r").field("data"))
    )
    assert(
      get(r, mem.field("r").field("data").index(1)) ==
        Set(m.ref("mem__1").field("r").field("data"))
    )

    val renameCount = r.underlying.map(_._2.size).sum
    assert(renameCount == 8, "it is enough to rename *to* 8 different signals")
    assert(r.underlying.size == 7, "it is enough to rename (from) 7 different signals")
  }

}

private object LowerTypesSpecUtils {
  private val typedCompiler = new TransformManager(Seq(Dependency(InferTypes)))
  def parseType(tpe: String): firrtl.ir.Type = {
    val src =
      s"""circuit c:
         |  module c:
         |    input c: $tpe
         |""".stripMargin
    val c = CircuitState(firrtl.Parser.parse(src), Seq())
    typedCompiler.execute(c).circuit.modules.head.ports.head.tpe
  }
  case class DestructResult(fields: Seq[String], renameMap: RenameMap)
  def destruct(n: String, tpe: String, namespace: Set[String]): DestructResult = {
    val ref = firrtl.ir.Field(n, firrtl.ir.Default, parseType(tpe))
    val renames = RenameMap()
    val mutableSet = scala.collection.mutable.HashSet[String]() ++ namespace
    val res = DestructTypes.destruct(m, ref, mutableSet, renames, Set())
    DestructResult(resultToFieldSeq(res), renames)
  }
  def resultToFieldSeq(res: Seq[(firrtl.ir.Field, String)]): Seq[String] =
    res.map(_._1).map(r => s"${r.flip.serialize}${r.name} : ${r.tpe.serialize}")
  def get(r: RenameMap, m: IsMember): Set[IsMember] = r.get(m).get.toSet
  protected val m = CircuitTarget("m").module("m")
}
