# Serial Port (UART)

This is a minimalistic version of a serial port, also called UART.

The main feature of this UART is that it contains just the bare bone
what is needed and therefore is easy to use and requires minimal resources.
Examples how to use it are part of the source in ```Uart.scala```

Testers are executed as follows and test the transmitter and
receiver:

```sbt "integrationTests/testOnly chisel3.std.uart.UartTxTests"```

```sbt "integrationTests/testOnly chisel3.std.uart.UartRxTests"```

Status: working in hardware (FPGA), documented in the Chisel book.
