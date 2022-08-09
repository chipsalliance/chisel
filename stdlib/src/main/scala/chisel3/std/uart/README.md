# Serial Port (UART)

This is a minimalistic version of a serial port, also called UART.

Testers are executed as follows and test the transmitter and
receiver:

```sbt "integrationTests/testOnly chisel3.std.uart.UartTxTests"```

```sbt "integrationTests/testOnly chisel3.std.uart.UartRxTests"```

Status: working in hardware (FPGA)
