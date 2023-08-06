# REQUIRES: bindings_python
# REQUIRES: capnp
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import esi

from circt.ir import *
from circt import passmanager

with Context() as ctx:
  circt.register_dialects(ctx)
  any_type = esi.AnyType.get()
  print(any_type)
  print()
  # CHECK: !esi.any

  list_type = esi.ListType.get(IntegerType.get_signless(3))
  print(list_type)
  print()
  # CHECK: !esi.list<i3>


# CHECK-LABEL: === testGen called with op:
# CHECK:       %0:2 = esi.service.impl_req svc @HostComms impl as "test"(%clk) : (i1) -> (i8, !esi.channel<i8>) {
# CHECK:         %2 = esi.service.req.to_client <@HostComms::@Recv>(["m1", "loopback_tohw"]) : !esi.channel<i8>
# CHECK:         esi.service.req.to_server %m1.loopback_fromhw -> <@HostComms::@Send>(["m1", "loopback_fromhw"]) : !esi.channel<i8>
def testGen(reqOp: esi.ServiceImplementReqOp) -> bool:
  print("=== testGen called with op:")
  reqOp.print()
  print()
  return True


esi.registerServiceGenerator("test", testGen)

with Context() as ctx:
  circt.register_dialects(ctx)
  mod = Module.parse("""
esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
}

msft.module @MsTop {} (%clk: i1) -> (chksum: i8) {
  %c = esi.service.instance svc @HostComms impl as  "test" (%clk) : (i1) -> (i8)
  msft.instance @m1 @MsLoopback (%clk) : (i1) -> ()
  msft.output %c : i8
}

msft.module @MsLoopback {} (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i8>
  msft.output
}
""")
  pm = passmanager.PassManager.parse("builtin.module(esi-connect-services)")
  pm.run(mod.operation)
