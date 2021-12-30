<!Doctype html>
<html>
<title> Two side column </title>
<body>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <table border ="0">
        <h1>Creating a Module</h1>
        <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>

```
    module foo (
                    input  a,
                    output b
                )
                assign b = a;
            endmodule
```

</td>
    <td>

```scala mdoc
    class Foo extends Module {
    val a = Input(Bool())
    val b = Output(Bool())
    b := a
    }
```

</td>
    <td>Text comes here</td>
         </tr>
    </table>
</body>
</html>

#Parameterizing a Module

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>

```
module ParameterizedWidthAdder(

input [in0Width-1:0] in0,
input [in1Width-1:0] in1,
output [sumWidth-1:0] sum
);
parameter in0Width = 8;
parameter in1Width = 1;
parameter sumWidth = 9;

assign sum = in0 + in1;

endmodule
```

</td>
<td>

```
class ParameterizedWidthAdder(in0Width: Int, in1Width: Int, sumWidth: Int) extends Module {
  require(in0Width >= 0)
  require(in1Width >= 0)
  require(sumWidth >= 0)
  val io = IO(new Bundle {
    val in0 = Input(UInt(in0Width.W))
    val in1 = Input(UInt(in1Width.W))
    val sum = Output(UInt(sumWidth.W))
  })
  // a +& b includes the carry, a + b does not
  io.sum := io.in0 +& io.in1
}
```
</td>
             <td>Text comes here</td>
</tr>
<tr>
<td>

```
module TestBench;
wire [31:0] sum;
ParameterizedWidthAdder  #(32, 32, 32) my32BitAdderWithTruncation (32'b0, 32'b0, sum);
endmodule
```
</td>
            <td>

```
val my32BitAdderWithTruncation = Module(new ParameterizedWidthAdder(32, 32, 32)
```
</td>
             <td>Text comes here</td>
         </tr>
    </table>
<html>
<body>

#Wire assignment

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>Text comes here</td>
            <td>chisel code comes here</td>
             <td>Text comes here</td>
         </tr>
    </table>
<html>
<body>

#Register assignment

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>Text comes here</td>
            <td>chisel code comes here</td>
             <td>Text comes here</td>
         </tr>
    </table>
<html>
<body>

#2-D assignment

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>Text comes here</td>
            <td>chisel code comes here</td>
             <td>Text comes here</td>
         </tr>
    </table>
<html>
<body>