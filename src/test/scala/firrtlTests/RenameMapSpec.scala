// See LICENSE for license details.

package firrtlTests

import firrtl.RenameMap
import firrtl.FIRRTLException
import firrtl.annotations.{
  Named,
  CircuitName,
  ModuleName,
  ComponentName
}

class RenameMapSpec extends FirrtlFlatSpec {
  val cir = CircuitName("Top")
  val cir2 = CircuitName("Pot")
  val cir3 = CircuitName("Cir3")
  val modA = ModuleName("A", cir)
  val modA2 = ModuleName("A", cir2)
  val modB = ModuleName("B", cir)
  val foo = ComponentName("foo", modA)
  val foo2 = ComponentName("foo", modA2)
  val bar = ComponentName("bar", modA)
  val fizz = ComponentName("fizz", modA)
  val fooB = ComponentName("foo", modB)
  val barB = ComponentName("bar", modB)

  behavior of "RenameMap"

  it should "return None if it does not rename something" in {
    val renames = RenameMap()
    renames.get(modA) should be (None)
    renames.get(foo) should be (None)
  }

  it should "return a Seq of renamed things if it does rename something" in {
    val renames = RenameMap()
    renames.rename(foo, bar)
    renames.get(foo) should be (Some(Seq(bar)))
  }

  it should "allow something to be renamed to multiple things" in {
    val renames = RenameMap()
    renames.rename(foo, bar)
    renames.rename(foo, fizz)
    renames.get(foo) should be (Some(Seq(bar, fizz)))
  }

  it should "allow something to be renamed to nothing (ie. deleted)" in {
    val renames = RenameMap()
    renames.rename(foo, Seq())
    renames.get(foo) should be (Some(Seq()))
  }

  it should "return None if something is renamed to itself" in {
    val renames = RenameMap()
    renames.rename(foo, foo)
    renames.get(foo) should be (None)
  }

  it should "allow components to change module" in {
    val renames = RenameMap()
    renames.rename(foo, fooB)
    renames.get(foo) should be (Some(Seq(fooB)))
  }

  it should "rename components if their module is renamed" in {
    val renames = RenameMap()
    renames.rename(modA, modB)
    renames.get(foo) should be (Some(Seq(fooB)))
    renames.get(bar) should be (Some(Seq(barB)))
  }

  it should "rename renamed components if the module of the target component is renamed" in {
    val renames = RenameMap()
    renames.rename(modA, modB)
    renames.rename(foo, bar)
    renames.get(foo) should be (Some(Seq(barB)))
  }

  it should "rename modules if their circuit is renamed" in {
    val renames = RenameMap()
    renames.rename(cir, cir2)
    renames.get(modA) should be (Some(Seq(modA2)))
  }

  it should "rename components if their circuit is renamed" in {
    val renames = RenameMap()
    renames.rename(cir, cir2)
    renames.get(foo) should be (Some(Seq(foo2)))
  }

  // Renaming `from` to each of the `tos` at the same time should error
  case class BadRename(from: Named, tos: Seq[Named])
  val badRenames =
    Seq(BadRename(foo, Seq(cir)),
        BadRename(foo, Seq(modA)),
        BadRename(modA, Seq(foo)),
        BadRename(modA, Seq(cir)),
        BadRename(cir, Seq(foo)),
        BadRename(cir, Seq(modA)),
        BadRename(cir, Seq(cir2, cir3))
       )
  // Run all BadRename tests
  for (BadRename(from, tos) <- badRenames) {
    val fromN = from.getClass.getSimpleName
    val tosN = tos.map(_.getClass.getSimpleName).mkString(", ")
    it should s"error if a $fromN is renamed to $tosN" in {
      val renames = RenameMap()
      for (to <- tos) { renames.rename(from, to) }
      a [FIRRTLException] shouldBe thrownBy {
        renames.get(foo)
      }
    }
  }
}
