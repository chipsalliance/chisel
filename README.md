chisel3 is a new FIRRTL based chisel

the current backward incompatiabilities with chisel 2.x are:

val wire = Bits(width = 15)

is

val wire = Wire(Bits(width = 15))
