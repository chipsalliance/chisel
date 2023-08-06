//===- Cosim_DpiPkg.sv - ESI cosim DPI declarations ---------*- verilog -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Package: Cosim_DpiPkg
//
// DPI-exposed funcs for cosimserver cosimulation unit-test.
//
//===----------------------------------------------------------------------===//

package Cosim_DpiPkg;

// --------------------- Cosim RPC Server --------------------------------------

// Start cosimserver (spawns server for HW-initiated work, listens for
// connections from new SW-clients).
import "DPI-C" sv2cCosimserverInit = function int cosim_init();

// Teardown cosimserver (disconnects from primary server port, stops connections
// from active clients).
import "DPI-C" sv2cCosimserverFinish = function void cosim_finish();

// --------------------- Endpoint Management -----------------------------------

// Register simulated device endpoints.
// - return 0 on success, non-zero on failure (duplicate EP registered).
import "DPI-C" sv2cCosimserverEpRegister =
  function int cosim_ep_register(
    // The endpoint ID.
    input string endpoint_id,
    // The capnp type id which the _RPC client_ is sending us.
    input longint send_type_id,
    // The send types max size, in bytes.
    input int send_type_size,
    // The capnp type id which we are sending to the _RPC client_.
    input longint esi_recv_type_id,
    // The recv types max size, in bytes.
    input int recv_type_size);

// --------------------- Endpoint Accessors ------------------------------------

// Attempt to send data to a client.
// - return 0 on success, negative on failure (unregistered EP).
import "DPI-C" sv2cCosimserverEpTryPut =
  function int cosim_ep_tryput(
    // The ID of the endpoint to which the data should be sent.
    input string endpoint_id,
    // A data buffer.
    input byte unsigned data[],
    // (Optional) Size of the buffer. If negative, will be dynamically detected.
    input int data_size = -1
    );

// Attempt to recieve data from a client.
//   - Returns negative when call failed (e.g. EP not registered).
//   - If no message, return 0 with size_bytes == 0.
//   - Assumes buffer is large enough to contain entire message. Fails if not
//   large enough. (In the future, will add support for getting the message into
//   a fixed-size buffer over multiple calls.)
import "DPI-C" sv2cCosimserverEpTryGet =
  function int cosim_ep_tryget(
    // The ID of the endpoint from which data should be recieved.
    input string endpoint_id,
    // The buffer in which to put the data. This should be 'output', but the
    // open source simulator Verilator doesn't seem to have a way to do this.
    inout byte unsigned data[],
    // Input: indicates the size of the data[] buffer. If -1, dynamically detect
    // size.
    // Output: the size of the message.
    inout  int unsigned data_size
    );

endpackage // Cosim_DpiPkg
