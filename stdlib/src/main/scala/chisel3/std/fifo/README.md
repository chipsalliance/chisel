# First-In-First-Out (FIFO) Queues

Different implementations of FIFO queues. Small queues can be register based
(```BubbleFifo```, ```DoubleBufferFifo```, or ```RegFifo```). Larger queues shall use on-chip
memory (```MemFifo```). Combination of two different FIFOs is shown in ```CombFifo```.

Testers are executed as follows:

``` sbt "testOnly chisel.lib.fifo.FifoSpec"```