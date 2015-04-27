#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <map>
#include <queue>
#include <ctime>
#include <vpi_user.h>

using namespace std;

/*==========================================================================
                 User Functions
=============================================================================*/
int32_t wire_poke_calltf(char *user_data) {
  queue<vpiHandle> modules;
  vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
  vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle);
  // First argument: <node_name>
  s_vpi_value node_s;
  node_s.format = vpiStringVal;
  vpi_get_value(vpi_scan(arg_iter), &node_s);
  // Second argument: <value>
  s_vpi_value value_s;
  value_s.format = vpiIntVal;
  vpi_get_value(vpi_scan(arg_iter), &value_s);
  vpi_free_object(arg_iter);

  vpiHandle test_handle = vpi_scan(vpi_iterate(vpiModule, NULL));
  vpiHandle top_handle = vpi_scan(vpi_iterate(vpiModule, test_handle));
  // Construct node paths
  string testname = vpi_get_str(vpiDefName, test_handle);
  istringstream iss(node_s.value.str);
  string nodename;
  iss >> nodename;
  ostringstream oss;
  oss << testname << "." << nodename;
  string nodepath = oss.str();

  // Examine the regs in the testbench
  // in order to give the correct input values
  string modulepath = vpi_get_str(vpiFullName, top_handle);
  string testpath = testname + nodepath.substr(modulepath.length(), nodepath.length() - modulepath.length());
  vpiHandle reg_iter = vpi_iterate(vpiReg, test_handle);
  while (vpiHandle reg_handle = vpi_scan(reg_iter)) {
    if (testpath == vpi_get_str(vpiFullName, reg_handle)) {
      vpi_put_value(reg_handle, &value_s, NULL, vpiNoDelay);
      vpi_printf("ok\n");
      return 0;
    }
  }

  // Start from the top module
  modules.push(top_handle);

  bool found = false;
  while (!modules.empty()) {
    vpiHandle mod_handle = modules.front();
    modules.pop();

    // Iterate its net
    vpiHandle net_iter = vpi_iterate(vpiNet, mod_handle);
    while (vpiHandle net_handle = vpi_scan(net_iter)) {
      if (nodepath == vpi_get_str(vpiFullName, net_handle)) {
        vpi_put_value(net_handle, &value_s, NULL, vpiNoDelay);
        found = true;
      }
      if (found) break;
    }
    if (found) break;

    // Iterate its reg
    vpiHandle reg_iter = vpi_iterate(vpiReg, mod_handle);
    while (vpiHandle reg_handle = vpi_scan(reg_iter)) {
      if (nodepath == vpi_get_str(vpiFullName, reg_handle)) {
        vpi_put_value(reg_handle, &value_s, NULL, vpiNoDelay);
        found = true;
      }
      if (found) break;
    }
    if (found) break;

    vpiHandle sub_iter = vpi_iterate(vpiModule, mod_handle);
    while (vpiHandle sub_handle = vpi_scan(sub_iter)) {
      modules.push(sub_handle);
    }
  }

  if (found)
    vpi_printf("ok\n");
  else
    vpi_printf("error\n");

  return 0;
}

int32_t wire_peek_calltf(char *user_data) {
  queue<vpiHandle> modules;
  vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
  vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle);
  // First argument: <node_name>
  s_vpi_value node_s;
  node_s.format = vpiStringVal;
  vpi_get_value(vpi_scan(arg_iter), &node_s);
  vpi_free_object(arg_iter);

  vpiHandle test_handle = vpi_scan(vpi_iterate(vpiModule, NULL));
  vpiHandle top_handle = vpi_scan(vpi_iterate(vpiModule, test_handle));
  // Construct node paths
  string testname = vpi_get_str(vpiDefName, test_handle);
  istringstream iss(node_s.value.str);
  string nodename;
  iss >> nodename;
  ostringstream oss;
  oss << testname << "." << nodename;
  string nodepath = oss.str();
  // Start from the top module
  modules.push(top_handle);

  s_vpi_value value_s;
  value_s.format = vpiHexStrVal;
  bool found = false;
  while (!modules.empty()) {
    vpiHandle mod_handle = modules.front();
    modules.pop();

    // Iterate its net
    vpiHandle net_iter = vpi_iterate(vpiNet, mod_handle);
    while (vpiHandle net_handle = vpi_scan(net_iter)) {
      if (nodepath == vpi_get_str(vpiFullName, net_handle)) {
        vpi_get_value(net_handle, &value_s);
        found = true;
      }
    }
    if (found) break;

    // Iterate its reg
    vpiHandle reg_iter = vpi_iterate(vpiReg, mod_handle);
    while (vpiHandle reg_handle = vpi_scan(reg_iter)) {
      if (nodepath == vpi_get_str(vpiFullName, reg_handle)) {
        vpi_get_value(reg_handle, &value_s);
        found = true;
      }
    }
    if (found) break;

    vpiHandle sub_iter = vpi_iterate(vpiModule, mod_handle);
    while (vpiHandle sub_handle = vpi_scan(sub_iter)) {
      modules.push(sub_handle);
    }
  }

  if (found)
    vpi_printf("0x%s\n", value_s.value.str);
  else
    vpi_printf("error\n");

  return 0;
}

int32_t mem_poke_calltf(char *user_data) {
  queue<vpiHandle> modules;
  vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
  vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle);
  // First argument: <mem_name>
  s_vpi_value node_s;
  node_s.format = vpiStringVal;
  vpi_get_value(vpi_scan(arg_iter), &node_s);
  // Second argument: <mem_index>
  s_vpi_value index_s;
  index_s.format = vpiIntVal;
  vpi_get_value(vpi_scan(arg_iter), &index_s);
  // Third argument: <value>
  s_vpi_value value_s;
  value_s.format = vpiIntVal;
  vpi_get_value(vpi_scan(arg_iter), &value_s);

  vpiHandle test_handle = vpi_scan(vpi_iterate(vpiModule, NULL));
  vpiHandle top_handle = vpi_scan(vpi_iterate(vpiModule, test_handle));
  // Construct node paths
  string testname = vpi_get_str(vpiDefName, test_handle);
  istringstream iss(node_s.value.str);
  string nodename;
  iss >> nodename;
  ostringstream oss;
  oss << testname << "." << nodename;
  string nodepath = oss.str();
  oss << "[" << index_s.value.integer << "]";
  string elmpath = oss.str();
  
  // Examine the reg arrays in the testbench
  // in order to give the correct input values
  string modulepath = vpi_get_str(vpiFullName, top_handle);
  string testpath = testname + nodepath.substr(modulepath.length(), nodepath.length() - modulepath.length());
  vpiHandle reg_iter = vpi_iterate(vpiRegArray, test_handle);
  while (vpiHandle reg_handle = vpi_scan(reg_iter)) {
    if (testpath == vpi_get_str(vpiFullName, reg_handle)) {
      vpiHandle elm_iter = vpi_iterate(vpiReg, reg_handle);
      while (vpiHandle elm_handle = vpi_scan(elm_iter)) {
          if (elmpath == vpi_get_str(vpiFullName, elm_handle)) {
          vpi_put_value(elm_handle, &value_s, NULL, vpiNoDelay);
          vpi_printf("ok\n");
          return 0;
        }
      }
    }
  }

  // Start from the top module
  modules.push(top_handle);

  bool found = false;
  while (!modules.empty()) {
    vpiHandle mod_handle = modules.front();
    modules.pop();

    // Iterate its net arrays
    vpiHandle net_iter = vpi_iterate(vpiNetArray, mod_handle);
    while (vpiHandle net_handle = vpi_scan(net_iter)) {
      if (nodepath == vpi_get_str(vpiFullName, net_handle)) {
        vpiHandle elm_iter = vpi_iterate(vpiNet, net_handle);
        while (vpiHandle elm_handle = vpi_scan(elm_iter)) {
          if (elmpath == vpi_get_str(vpiFullName, elm_handle)){
            vpi_put_value(elm_handle, &value_s, NULL, vpiNoDelay);
            found = true;
          }
          if (found) break;
        }
      }
      if (found) break;
    }
    if (found) break;

    // Iterate its reg arrays
    vpiHandle reg_iter = vpi_iterate(vpiRegArray, mod_handle);
    while (vpiHandle reg_handle = vpi_scan(reg_iter)) {
      if (nodepath == vpi_get_str(vpiFullName, reg_handle)) {
        vpiHandle elm_iter = vpi_iterate(vpiReg, reg_handle);
        while (vpiHandle elm_handle = vpi_scan(elm_iter)) {
          if (elmpath == vpi_get_str(vpiFullName, elm_handle)){
            vpi_put_value(elm_handle, &value_s, NULL, vpiNoDelay);
            found = true;
          }
          if (found) break;
        }
      }
      if (found) break;
    }
    if (found) break;

    vpiHandle sub_iter = vpi_iterate(vpiModule, mod_handle);
    while (vpiHandle sub_handle = vpi_scan(sub_iter)) {
      modules.push(sub_handle);
    }
  }

  if (found)
    vpi_printf("ok\n");
  else
    vpi_printf("error\n");
}

int32_t mem_peek_calltf(char *user_data) {
  queue<vpiHandle> modules;
  vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
  vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle);
  // First argument: <node_name>
  s_vpi_value node_s;
  node_s.format = vpiStringVal;
  vpi_get_value(vpi_scan(arg_iter), &node_s);
  // Second argument: <mem_index>
  s_vpi_value index_s;
  index_s.format = vpiIntVal;
  vpi_get_value(vpi_scan(arg_iter), &index_s);

  vpiHandle test_handle = vpi_scan(vpi_iterate(vpiModule, NULL));
  vpiHandle top_handle = vpi_scan(vpi_iterate(vpiModule, test_handle));
  // Construct node paths
  string testname = vpi_get_str(vpiDefName, test_handle);
  istringstream iss(node_s.value.str);
  string nodename;
  iss >> nodename;
  ostringstream oss;
  oss << testname << "." << nodename;
  string nodepath = oss.str();
  oss << "[" << index_s.value.integer << "]";
  string elmpath = oss.str();
  // Start from the top module
  modules.push(top_handle);

  s_vpi_value value_s;
  value_s.format = vpiHexStrVal;
  bool found = false;
  while (!modules.empty()) {
    vpiHandle mod_handle = modules.front();
    modules.pop();

    // Iterate its net arrays
    vpiHandle net_iter = vpi_iterate(vpiNetArray, mod_handle);
    while (vpiHandle net_handle = vpi_scan(net_iter)) {
      if (nodepath == vpi_get_str(vpiFullName, net_handle)) {
        vpiHandle elm_iter = vpi_iterate(vpiNet, net_handle);
        while (vpiHandle elm_handle = vpi_scan(elm_iter)) {
          if (elmpath == vpi_get_str(vpiFullName, elm_handle)){
            vpi_get_value(elm_handle, &value_s);
            found = true;
          }
          if (found) break;
        }
      }
      if (found) break;
    }
    if (found) break;

    // Iterate its reg arrays
    vpiHandle reg_iter = vpi_iterate(vpiRegArray, mod_handle);
    while (vpiHandle reg_handle = vpi_scan(reg_iter)) {
      if (nodepath == vpi_get_str(vpiFullName, reg_handle)) {
        vpiHandle elm_iter = vpi_iterate(vpiReg, reg_handle);
        while (vpiHandle elm_handle = vpi_scan(elm_iter)) {
          if (elmpath == vpi_get_str(vpiFullName, elm_handle)){
            vpi_get_value(elm_handle, &value_s);
            found = true;
          }
          if (found) break;
        }
      }
      if (found) break;
    }
    if (found) break;

    vpiHandle sub_iter = vpi_iterate(vpiModule, mod_handle);
    while (vpiHandle sub_handle = vpi_scan(sub_iter)) {
      modules.push(sub_handle);
    }
  }

  if (found)
    vpi_printf("0x%s\n", value_s.value.str);
  else
    vpi_printf("error\n");
}

/*==========================================================================
                 Compile Time Functions
=============================================================================*/

int32_t wire_poke_compiletf (char *user_data) {
  vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
  vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle), arg_handle;
  bool error = false;

  if (arg_iter == NULL) {
    vpi_printf("ERROR: $wire_poke requires at least two argument(nodename, hex_value)\n"); 
    vpi_control(vpiFinish, 1); /* abort simulation */
    return 0;
  }

  arg_handle = vpi_scan(arg_iter);
  if (vpi_get(vpiType, arg_handle) != vpiReg && vpi_get(vpiType, arg_handle) != vpiStringConst) {
    vpi_printf("ERROR: $wire_poke requires the first argument as a string(nodename)\n");
    error = true;
  }

  arg_handle = vpi_scan(arg_iter);
  if (vpi_get(vpiType, arg_handle) != vpiReg && vpi_get(vpiType, arg_handle) != vpiHexConst) {
    vpi_printf("ERROR: $wire_poke requires the second argument as a hex number(value)\n");
    error = true;
  }

  if (vpi_scan(arg_iter) != NULL) {
    vpi_printf("ERROR: $wire_poke requires only two arguments(nodename, hex_value))\n");
    error = true;
  }

  if (error) {
    vpi_control(vpiFinish, 1); /* abort simulation */
    return 0;
  }
  return 0; 
}

int32_t wire_peek_compiletf (char *user_data) {
  vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
  vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle), arg_handle;
  bool error = false;

  if (arg_iter == NULL) {
    vpi_printf("ERROR: $wire_peek requires at least one argument(nodename)\n"); 
    vpi_control(vpiFinish, 1); /* abort simulation */
    return 0;
  }

  arg_handle = vpi_scan(arg_iter);
  if (vpi_get(vpiType, arg_handle) != vpiReg && vpi_get(vpiType, arg_handle) != vpiStringConst) {
    vpi_printf("ERROR: $wire_peek requires the first argument as a string(nodename)\n");
    error = true;
  }

  if (vpi_scan(arg_iter) != NULL) {
    vpi_printf("ERROR: $wire_peek requires only one arguments(nodename))\n");
    error = true;
  }

  if (error) {
    vpi_control(vpiFinish, 1); /* abort simulation */
    return 0;
  }
  return 0; 
}

int32_t mem_poke_compiletf (char *user_data) {
  vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
  vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle), arg_handle;
  bool error = false;

  if (arg_iter == NULL) {
    vpi_printf("ERROR: $mem_poke requires at least three argument(nodename, offset, hex_value)\n"); 
    vpi_control(vpiFinish, 1); /* abort simulation */
    return 0;
  }

  arg_handle = vpi_scan(arg_iter);
  if (vpi_get(vpiType, arg_handle) != vpiReg && vpi_get(vpiType, arg_handle) != vpiStringConst) {
    vpi_printf("ERROR: $mem_poke requires the first argument as a string(nodename)\n");
    error = true;
  }

  arg_handle = vpi_scan(arg_iter);
  if (vpi_get(vpiType, arg_handle) != vpiIntegerVar && vpi_get(vpiType, arg_handle) != vpiDecConst) {
    vpi_printf("ERROR: $mem_poke requires the second argument as a integer(offset)\n");
    error = true;
  }

  arg_handle = vpi_scan(arg_iter);
  if (vpi_get(vpiType, arg_handle) != vpiReg && vpi_get(vpiType, arg_handle) != vpiHexConst) {
    vpi_printf("ERROR: $mem_poke requires the third argument as a hex string(value)\n");
    error = true;
  }

  if (vpi_scan(arg_iter) != NULL) {
    vpi_printf("ERROR: $mem_poke requires only three arguments(nodename, offset, hex_value))\n");
    error = true;
  }

  if (error) {
    vpi_control(vpiFinish, 1); /* abort simulation */
    return 0;
  }
  return 0; 
}
 
int32_t mem_peek_compiletf (char *user_data) {
  vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
  vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle), arg_handle;
  bool error = false;

  if (arg_iter == NULL) {
    vpi_printf("ERROR: $mem_peek requires at least two argument(nodename, offset, hex_value)\n"); 
    vpi_control(vpiFinish, 1); /* abort simulation */
    return 0;
  }

  arg_handle = vpi_scan(arg_iter);
  if (vpi_get(vpiType, arg_handle) != vpiReg && vpi_get(vpiType, arg_handle) != vpiStringConst) {
    vpi_printf("ERROR: $mem_peek requires the first argument as a string(nodename)\n");
    error = true;
  }

  arg_handle = vpi_scan(arg_iter);
  if (vpi_get(vpiType, arg_handle) != vpiIntegerVar && vpi_get(vpiType, arg_handle) != vpiDecConst) {
    vpi_printf("ERROR: $mem_peek requires the second argument as a integer(offset)\n");
    error = true;
  }

  if (vpi_scan(arg_iter) != NULL) {
    vpi_printf("ERROR: $mem_peek requires only two arguments(nodename, offset))\n");
    error = true;
  }

  if (error) {
    vpi_control(vpiFinish, 1); /* abort simulation */
    return 0;
  }
  return 0; 
}
 
 
/*==========================================================================
                 Registration Functions
=============================================================================*/

void wire_poke_registration() {
  s_vpi_systf_data tf_data;
  
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = "$wire_poke";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = wire_poke_calltf;
  tf_data.compiletf = wire_poke_compiletf;
  tf_data.user_data = NULL;

  vpi_register_systf(&tf_data);

  return;
}

void wire_peek_registration() {
  s_vpi_systf_data tf_data;
  
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = "$wire_peek";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = wire_peek_calltf;
  tf_data.compiletf = wire_peek_compiletf;
  tf_data.user_data = NULL;

  vpi_register_systf(&tf_data);

  return;
}

void mem_poke_registration() {
  s_vpi_systf_data tf_data;
  
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = "$mem_poke";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = mem_poke_calltf;
  tf_data.compiletf = mem_poke_compiletf;
  tf_data.user_data = NULL;

  vpi_register_systf(&tf_data);

  return;
}

void mem_peek_registration() {
  s_vpi_systf_data tf_data;
  
  tf_data.type      = vpiSysTask;
  tf_data.tfname    = "$mem_peek";
  tf_data.sizetf    = NULL;
  tf_data.calltf    = mem_peek_calltf;
  tf_data.compiletf = mem_peek_compiletf;
  tf_data.user_data = NULL;

  vpi_register_systf(&tf_data);

  return;
}

/*==========================================================================
                 Start-up Array
=============================================================================*/
void (*vlog_startup_routines[]) () = {
  wire_poke_registration,
  wire_peek_registration,
  mem_poke_registration,
  mem_peek_registration,
  0
};
