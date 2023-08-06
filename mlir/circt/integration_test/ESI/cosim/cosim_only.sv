// REQUIRES: esi-cosim
// RUN: esi-cosim-runner.py %s %s
// PY: import loopback as test
// PY: rpc = test.LoopbackTester(rpcschemapath, simhostport)
// PY: print(rpc.cosim.list().wait())
// PY: rpc.test_list()
// PY: rpc.test_open_close()
// PY: rpc.test_3bytes(5)

import Cosim_DpiPkg::*;

module top(
    input logic clk,
    input logic rst
);
    localparam int TYPE_SIZE_BITS =
        (1 * 64) + // root message
        (1 * 64) + // list header
        (1 * 64);  // list of length 3 bytes, rounded up to multiples of 8 bytes

    wire DataOutValid;
    wire DataOutReady = 1;
    wire [TYPE_SIZE_BITS-1:0] DataOut;
    
    logic DataInValid;
    logic DataInReady;
    logic [TYPE_SIZE_BITS-1:0] DataIn;

    Cosim_Endpoint #(
        .RECV_TYPE_ID(1),
        .RECV_TYPE_SIZE_BITS(TYPE_SIZE_BITS),
        .SEND_TYPE_ID(1),
        .SEND_TYPE_SIZE_BITS(TYPE_SIZE_BITS)
    ) ep (
        .*
    );

    always@(posedge clk)
    begin
        if (~rst)
        begin
            if (DataOutValid && DataOutReady)
            begin
                $display("[%d] Recv'd: %h", $time(), DataOut);
                DataIn <= DataOut;
            end
            DataInValid <= DataOutValid && DataOutReady;

            if (DataInValid && DataInReady)
            begin
                $display("[%d] Sent: %h", $time(), DataIn);
            end
        end
        else
        begin
            DataInValid <= 0;
        end
    end
endmodule
