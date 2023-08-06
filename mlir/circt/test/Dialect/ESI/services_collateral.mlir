// REQUIRES: capnp
// RUN: circt-opt %s --esi-connect-services --esi-emit-collateral=tops=LoopbackCosimTopWrapper,LoopbackCosimBSP --esi-clean-metadata --lower-msft-to-hw --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw  --export-verilog -o %t1.mlir  | FileCheck %s

// CHECK-LABEL: // ----- 8< ----- FILE "services.json" ----- 8< -----

// CHECK-LABEL: "declarations": [ 

// CHECK-LABEL:    "name": "HostComms",
// CHECK-NEXT:     "ports": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "name": "Send",
// CHECK-NEXT:           "to-server-type": {
// CHECK-NEXT:             "dialect": "esi",
// CHECK-NEXT:             "inner": {
// CHECK-NEXT:               "dialect": "esi",
// CHECK-NEXT:               "mnemonic": "any"
// CHECK-NEXT:             },
// CHECK-NEXT:             "mnemonic": "channel"
// CHECK-NEXT:           }
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "name": "Recv",
// CHECK-NEXT:           "to-client-type": {
// CHECK-NEXT:             "dialect": "esi",
// CHECK-NEXT:             "inner": {
// CHECK-NEXT:               "dialect": "builtin",
// CHECK-NEXT:               "mnemonic": "i8"
// CHECK-NEXT:             },
// CHECK-NEXT:             "mnemonic": "channel"
// CHECK-NEXT:           }
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "name": "ReqResp",
// CHECK-NEXT:           "to-client-type": {
// CHECK-NEXT:             "dialect": "esi",
// CHECK-NEXT:             "inner": {
// CHECK-NEXT:               "dialect": "builtin",
// CHECK-NEXT:               "mnemonic": "i16"
// CHECK-NEXT:             },
// CHECK-NEXT:             "mnemonic": "channel"
// CHECK-NEXT:           },
// CHECK-NEXT:           "to-server-type": {
// CHECK-NEXT:             "dialect": "esi",
// CHECK-NEXT:             "inner": {
// CHECK-NEXT:               "dialect": "builtin",
// CHECK-NEXT:               "mnemonic": "i8"
// CHECK-NEXT:             },
// CHECK-NEXT:             "mnemonic": "channel"
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       ]

// CHECK-LABEL:     "name": "BSP",
// CHECK-NEXT:      "ports": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "name": "Send",
// CHECK-NEXT:          "to-server-type": {
// CHECK-NEXT:            "dialect": "esi",
// CHECK-NEXT:            "inner": {
// CHECK-NEXT:              "dialect": "hw",
// CHECK-NEXT:              "fields": [
// CHECK-NEXT:                {
// CHECK-NEXT:                  "name": "foo",
// CHECK-NEXT:                  "type": {
// CHECK-NEXT:                    "dialect": "builtin",
// CHECK-NEXT:                    "mnemonic": "i3"
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              ],
// CHECK-NEXT:              "mnemonic": "struct"
// CHECK-NEXT:            },
// CHECK-NEXT:            "mnemonic": "channel"
// CHECK-NEXT:          }
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "name": "ReqResp",
// CHECK-NEXT:          "to-client-type": {
// CHECK-NEXT:            "dialect": "esi",
// CHECK-NEXT:            "inner": {
// CHECK-NEXT:              "dialect": "builtin",
// CHECK-NEXT:              "mnemonic": "i16"
// CHECK-NEXT:            },
// CHECK-NEXT:            "mnemonic": "channel"
// CHECK-NEXT:          },
// CHECK-NEXT:          "to-server-type": {
// CHECK-NEXT:            "dialect": "esi",
// CHECK-NEXT:            "inner": {
// CHECK-NEXT:              "dialect": "builtin",
// CHECK-NEXT:              "mnemonic": "i8"
// CHECK-NEXT:            },
// CHECK-NEXT:            "mnemonic": "channel"
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    }

// CHECK-LABEL: "top_levels": [

// CHECK-LABEL: "module": "@LoopbackCosimTopWrapper",
// CHECK-NEXT:  "services": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "service": "HostComms",
// CHECK-NEXT:      "instance_path": [
// CHECK-NEXT:        {
// CHECK-NEXT:        "inner": "top",
// CHECK-NEXT:        "outer_sym": "LoopbackCosimTop"
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-LABEL: "module": "@LoopbackCosimBSP",
// CHECK-NEXT:   "services": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "service": "BSP",
// CHECK-NEXT:       "instance_path": []
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

// CHECK-LABEL:  "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "symbol": "LoopbackCosimTop",
// CHECK-NEXT:       "services": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "service": "HostComms",
// CHECK-NEXT:           "impl_type": "cosim",
// CHECK-NEXT:           "clients": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "client_name": [
// CHECK-NEXT:                 "m1",
// CHECK-NEXT:                 "loopback_inout"
// CHECK-NEXT:               ],
// CHECK-NEXT:               "port": {
// CHECK-NEXT:                 "inner": "ReqResp",
// CHECK-NEXT:                 "outer_sym": "HostComms"
// CHECK-NEXT:               },
// CHECK-NEXT:               "to_client_type": {
// CHECK-NEXT:                 "capnp_name": "I16",
// CHECK-NEXT:                 "capnp_type_id": 15002640976408729367,
// CHECK-NEXT:                 "hw_bitwidth": 16,
// CHECK-NEXT:                 "mlir_name": "!esi.channel<i16>",
// CHECK-NEXT:                 "type_desc": {
// CHECK-NEXT:                   "dialect": "esi",
// CHECK-NEXT:                   "inner": {
// CHECK-NEXT:                     "dialect": "builtin",
// CHECK-NEXT:                     "mnemonic": "i16"
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "mnemonic": "channel"
// CHECK-NEXT:                 }
// CHECK-NEXT:               },
// CHECK-NEXT:               "to_server_type": {
// CHECK-NEXT:                 "capnp_name": "I8",
// CHECK-NEXT:                 "capnp_type_id": 9950424317211852587,
// CHECK-NEXT:                 "hw_bitwidth": 8,
// CHECK-NEXT:                 "mlir_name": "!esi.channel<i8>",
// CHECK-NEXT:                 "type_desc": {
// CHECK-NEXT:                   "dialect": "esi",
// CHECK-NEXT:                   "inner": {
// CHECK-NEXT:                     "dialect": "builtin",
// CHECK-NEXT:                     "mnemonic": "i8"
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "mnemonic": "channel"
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:             {
// CHECK-NEXT:               "client_name": [
// CHECK-NEXT:                 "m2"
// CHECK-NEXT:               ],
// CHECK-NEXT:               "port": {
// CHECK-NEXT:                 "inner": "Send",
// CHECK-NEXT:                 "outer_sym": "HostComms"
// CHECK-NEXT:               },
// CHECK-NEXT:               "to_server_type": {
// CHECK-NEXT:                 "capnp_name": "Struct12633565592854986228",
// CHECK-NEXT:                 "capnp_type_id": 12633565592854986228,
// CHECK-NEXT:                 "hw_bitwidth": 3,
// CHECK-NEXT:                 "mlir_name": "!esi.channel<!hw.struct<foo: i3>>",
// CHECK-NEXT:                 "type_desc": {
// CHECK-NEXT:                   "dialect": "esi",
// CHECK-NEXT:                   "inner": {
// CHECK-NEXT:                     "dialect": "hw",
// CHECK-NEXT:                     "fields": [
// CHECK-NEXT:                       {
// CHECK-NEXT:                         "name": "foo",
// CHECK-NEXT:                         "type": {
// CHECK-NEXT:                           "dialect": "builtin",
// CHECK-NEXT:                           "mnemonic": "i3"
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     ],
// CHECK-NEXT:                     "mnemonic": "struct"
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "mnemonic": "channel"
// CHECK-NEXT:                 }
// CHECK-NEXT:               }

// CHECK-LABEL:  "symbol": "LoopbackCosimBSP",
// CHECK-NEXT:   "services": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "service": "BSP",
// CHECK-NEXT:       "impl_type": "cosim",
// CHECK-NEXT:       "clients": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "client_name": [
// CHECK-NEXT:             "m1",
// CHECK-NEXT:             "loopback_inout"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "port": {
// CHECK-NEXT:             "inner": "ReqResp",
// CHECK-NEXT:             "outer_sym": "HostComms"
// CHECK-NEXT:           },
// CHECK-NEXT:           "to_client_type": {
// CHECK-NEXT:             "capnp_name": "I16",
// CHECK-NEXT:             "capnp_type_id": 15002640976408729367,
// CHECK-NEXT:             "hw_bitwidth": 16,
// CHECK-NEXT:             "mlir_name": "!esi.channel<i16>",
// CHECK-NEXT:             "type_desc": {
// CHECK-NEXT:               "dialect": "esi",
// CHECK-NEXT:               "inner": {
// CHECK-NEXT:                 "dialect": "builtin",
// CHECK-NEXT:                 "mnemonic": "i16"
// CHECK-NEXT:               },
// CHECK-NEXT:               "mnemonic": "channel"
// CHECK-NEXT:             }
// CHECK-NEXT:           },
// CHECK-NEXT:           "to_server_type": {
// CHECK-NEXT:             "capnp_name": "I8",
// CHECK-NEXT:             "capnp_type_id": 9950424317211852587,
// CHECK-NEXT:             "hw_bitwidth": 8,
// CHECK-NEXT:             "mlir_name": "!esi.channel<i8>",
// CHECK-NEXT:             "type_desc": {
// CHECK-NEXT:               "dialect": "esi",
// CHECK-NEXT:               "inner": {
// CHECK-NEXT:                 "dialect": "builtin",
// CHECK-NEXT:                 "mnemonic": "i8"
// CHECK-NEXT:               },
// CHECK-NEXT:               "mnemonic": "channel"
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "client_name": [
// CHECK-NEXT:             "m2"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "port": {
// CHECK-NEXT:             "inner": "Send",
// CHECK-NEXT:             "outer_sym": "HostComms"
// CHECK-NEXT:           },
// CHECK-NEXT:           "to_server_type": {
// CHECK-NEXT:             "capnp_name": "Struct12633565592854986228",
// CHECK-NEXT:             "capnp_type_id": 12633565592854986228,
// CHECK-NEXT:             "hw_bitwidth": 3,
// CHECK-NEXT:             "mlir_name": "!esi.channel<!hw.struct<foo: i3>>",
// CHECK-NEXT:             "type_desc": {
// CHECK-NEXT:               "dialect": "esi",
// CHECK-NEXT:               "inner": {
// CHECK-NEXT:                 "dialect": "hw",
// CHECK-NEXT:                 "fields": [
// CHECK-NEXT:                   {
// CHECK-NEXT:                     "name": "foo",
// CHECK-NEXT:                     "type": {
// CHECK-NEXT:                       "dialect": "builtin",
// CHECK-NEXT:                       "mnemonic": "i3"
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 ],
// CHECK-NEXT:                 "mnemonic": "struct"
// CHECK-NEXT:               },
// CHECK-NEXT:               "mnemonic": "channel"
// CHECK-NEXT:             }
// CHECK-NEXT:           }

esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
  esi.service.inout @ReqResp : !esi.channel<i8> -> !esi.channel<i16>
}

msft.module @SendStruct {} (%clk: i1) -> () {
  %c2_i3 = hw.constant 2 : i3
  %s = hw.struct_create (%c2_i3) : !hw.struct<foo: i3>
  %c1_i1 = hw.constant 1 : i1
  %schan, %ready = esi.wrap.vr %s, %c1_i1 : !hw.struct<foo: i3>
  esi.service.req.to_server %schan -> <@HostComms::@Send> ([]): !esi.channel<!hw.struct<foo: i3>>
  msft.output
}

msft.module @InOutLoopback {} (%clk: i1) -> () {
  %dataIn = esi.service.req.inout %dataTrunc -> <@HostComms::@ReqResp> (["loopback_inout"]) : !esi.channel<i8> -> !esi.channel<i16>
  %unwrap, %valid = esi.unwrap.vr %dataIn, %rdy: i16
  %trunc = comb.extract %unwrap from 0 : (i16) -> (i8)
  %dataTrunc, %rdy = esi.wrap.vr %trunc, %valid : i8
  msft.output
}

msft.module @LoopbackCosimTop {} (%clk: i1, %rst: i1) {
  esi.service.instance svc @HostComms impl as "cosim" (%clk, %rst) : (i1, i1) -> ()
  msft.instance @m1 @InOutLoopback(%clk) : (i1) -> ()
  msft.instance @m2 @SendStruct(%clk) : (i1) -> ()
  msft.output
}
msft.module @LoopbackCosimTopWrapper {} (%clk: i1, %rst: i1) {
  msft.instance @top @LoopbackCosimTop(%clk, %rst) : (i1, i1) -> ()
  msft.output
}

msft.module @LoopbackCosimBSP {} (%clk: i1, %rst: i1) {
  esi.service.instance impl as "cosim" (%clk, %rst) : (i1, i1) -> ()
  msft.instance @m1 @InOutLoopback(%clk) : (i1) -> ()
  msft.instance @m2 @SendStruct(%clk) : (i1) -> ()
  msft.output
}
