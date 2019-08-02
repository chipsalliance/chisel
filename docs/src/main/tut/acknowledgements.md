---
layout: page
title:  "Acknowledgements"
section: "Acknowledgements"
position: 16
---
Many people have helped out in the design of Chisel, and we thank them
for their patience, bravery, and belief in a better way.  Many
Berkeley EECS students in the Isis group gave weekly feedback as the
design evolved including but not limited to Yunsup Lee, Andrew
Waterman, Scott Beamer, Chris Celio, etc.  Yunsup Lee gave us feedback
in response to the first RISC-V implementation, called TrainWreck,
translated from Verilog to Chisel.  Andrew Waterman and Yunsup Lee
helped us get our Verilog backend up and running and Chisel TrainWreck
running on an FPGA.  Brian Richards was the first actual Chisel user,
first translating (with Huy Vo) John Hauser's FPU Verilog code to
Chisel, and later implementing generic memory blocks.  Brian gave many
invaluable comments on the design and brought a vast experience in
hardware design and design tools.  Chris Batten shared his fast
multiword C++ template library that inspired our fast emulation
library.  Huy Vo became our undergraduate research assistant and was
the first to actually assist in the Chisel implementation.  We
appreciate all the EECS students who participated in the Chisel
bootcamp and proposed and worked on hardware design projects all of
which pushed the Chisel envelope.  We appreciate the work that James
Martin and Alex Williams did in writing and translating network and
memory controllers and non-blocking caches.  Finally, Chisel's
functional programming and bit-width inference ideas were inspired by
earlier work on a hardware description language called *Gel* designed in
collaboration with Dany Qumsiyeh and Mark Tobenkin.

<!--- Who else? --->

Bachrach, J., Vo, H., Richards, B., Lee, Y., Waterman, A., Avizienis, Wawrzynek, J., Asanovic, K.
 **Chisel: Constructing Hardware in a Scala Embedded Language**.
in DAC '12.
Bachrach, J., Qumsiyeh, D., Tobenkin, M.
 **Hardware Scripting in Gel**.
in Field-Programmable Custom Computing Machines, 2008. FCCM '08. 16th.
