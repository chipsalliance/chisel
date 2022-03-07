package chisel3.std

import org.scalatest.flatspec.AnyFlatSpec

class AddressSetTests extends AnyFlatSpec {
  behavior.of("AddressSet")

  it should "check containment with bigint" in {
    // contiguous address set
    assert(AddressSet(0x200, 0xff).contiguous)
    assert(!AddressSet(0x200, 0xff).contains(0))
    assert(!AddressSet(0x200, 0xff).contains(0x300))
    assert(!AddressSet(0x200, 0xff).contains(0x200 - 1))
    assert(AddressSet(0x200, 0xff).contains(0x200))
    assert(AddressSet(0x200, 0xff).contains(0x2ff))
    assert(AddressSet(0x200, 0xff).contains(0x2ff))

    // non-contiguous address set
    assert(!AddressSet(0x1000, 0xf0f).contiguous)
    assert(!AddressSet(0x1000, 0xf0f).contains(0))
    assert(!AddressSet(0x1000, 0xf0f).contains(0x1000 - 1))
    assert(!AddressSet(0x1000, 0xf0f).contains(0x100f + 1))
    assert(!AddressSet(0x1000, 0xf0f).contains(0x1f00 - 1))
    assert(!AddressSet(0x1000, 0xf0f).contains(0x1f0f + 1))
    assert(AddressSet(0x1000, 0xf0f).contains(0x1000))
    assert(AddressSet(0x1000, 0xf0f).contains(0x100f))
    assert(AddressSet(0x1000, 0xf0f).contains(0x1f00))
    assert(AddressSet(0x1000, 0xf0f).contains(0x1f0f))
  }

}
