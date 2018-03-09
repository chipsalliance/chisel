#ifndef __VPI_H
#define __VPI_H

#include "vpi_user.h"
#include "sim_api.h"
#include <queue>

extern "C" PLI_INT32 tick_cb(p_cb_data cb_data);
extern "C" PLI_INT32 sim_start_cb(p_cb_data cb_data);
extern "C" PLI_INT32 sim_end_cb(p_cb_data cb_data);

class vpi_api_t: public sim_api_t<vpiHandle> {
public:
  vpi_api_t() { srand(time(NULL)); }
  ~vpi_api_t() { }

  virtual void tick() {
    while(!forces.empty()) {
      s_vpi_value value_s;
      value_s.format = vpiHexStrVal;
      vpi_get_value(forces.front(), &value_s);
      vpi_put_value(forces.front(), &value_s, NULL, vpiReleaseFlag);
      vpi_put_value(forces.front(), &value_s, NULL, vpiNoDelay);
      forces.pop();
    }
    sim_api_t::tick();
  }

  void init_rsts() {
    vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
    vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle);
    // Cache Resets
    while (vpiHandle arg_handle = vpi_scan(arg_iter)) {
      sim_data.resets.push_back(arg_handle);
    }
  }

  void init_ins() {
    vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
    vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle);
    // Cache Inputs  
    while (vpiHandle arg_handle = vpi_scan(arg_iter)) {
      sim_data.inputs.push_back(arg_handle);
    }
  }

  void init_outs() {
    vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
    vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle);
    // Cache Outputs
    while (vpiHandle arg_handle = vpi_scan(arg_iter)) {
      sim_data.outputs.push_back(arg_handle);
    }
  }

  void init_sigs() {
    vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
    vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle);
    top_handle = vpi_scan(arg_iter);
    dump_handle = vpi_scan(arg_iter);
    probe_signals();
  }

private:
  vpiHandle top_handle;
  vpiHandle dump_handle;
  std::queue<vpiHandle> forces;
  std::map<vpiHandle, size_t> sizes;
  std::map<vpiHandle, size_t> chunks;

  void put_value(vpiHandle& sig, std::string& value, bool force=false) {
    s_vpi_value value_s;
    value_s.format    = vpiHexStrVal;
    value_s.value.str = (PLI_BYTE8*) value.c_str();
    vpi_put_value(sig, &value_s, NULL, force ? vpiForceFlag : vpiNoDelay);
    if (force) forces.push(sig); 
  }

  size_t put_value(vpiHandle& sig, uint64_t* data, bool force=false) {
    size_t chunk = get_chunk(sig);
    s_vpi_value  value_s;
    s_vpi_vecval vecval_s[2*chunk];
    value_s.format       = vpiVectorVal;
    value_s.value.vector = vecval_s; 
    for (size_t i = 0 ; i < chunk ; i++) {
      value_s.value.vector[2*i].aval = (int)data[i]; 
      value_s.value.vector[2*i+1].aval = (int)(data[i]>>32);
      value_s.value.vector[2*i].bval = 0;
      value_s.value.vector[2*i+1].bval = 0;
    }
    vpi_put_value(sig, &value_s, NULL, force ? vpiForceFlag : vpiNoDelay);
    if (force) forces.push(sig); 
    return chunk;
  }

  inline std::string get_value(vpiHandle& sig) {
    s_vpi_value value_s;
    value_s.format = vpiHexStrVal;
    vpi_get_value(sig, &value_s);
    return value_s.value.str;
  }

  size_t get_value(vpiHandle& sig, uint64_t* data) {
    size_t size = get_size(sig);
    s_vpi_value  value_s;
    s_vpi_vecval vecval_s[size];
    value_s.format       = vpiVectorVal;
    value_s.value.vector = vecval_s;
    vpi_get_value(sig, &value_s);
    for (size_t i = 0 ; i < size ; i += 2) {
      data[i>>1] = (uint64_t)value_s.value.vector[i].aval;
    }
    for (size_t i = 1 ; i < size ; i += 2) {
      data[i>>1] |= (uint64_t)value_s.value.vector[i].aval << 32;
    }
    return ((size-1)>>1)+1;
  }

  inline size_t get_size(vpiHandle &sig) {
    return (size_t) (((vpi_get(vpiSize, sig)-1) >> 5) + 1);
  }

  inline size_t get_chunk(vpiHandle &sig) {
    return (size_t) (((vpi_get(vpiSize, sig)-1) >> 6) + 1);
  }

  virtual void reset() {
    for (size_t i = 0 ; i < sim_data.resets.size() ; i++) {
      s_vpi_value value_s;
      value_s.format    = vpiHexStrVal;
      value_s.value.str = (PLI_BYTE8*) "1";
      vpi_put_value(sim_data.resets[i], &value_s, NULL, vpiNoDelay);
    }
  }

  virtual void start() {
    for (size_t i = 0 ; i < sim_data.resets.size() ; i++) {
      s_vpi_value value_s;
      value_s.format    = vpiHexStrVal;
      value_s.value.str = (PLI_BYTE8*) "0";
      vpi_put_value(sim_data.resets[i], &value_s, NULL, vpiNoDelay);
    }
  }

  virtual void finish() {
    vpi_control(vpiFinish, 0);
  }

  virtual void step() { }

  virtual void update() {
    s_cb_data data_s;
    s_vpi_time time_s;
    time_s.type      = vpiSimTime;
    time_s.low       = 0;
    time_s.high      = 0;
    data_s.reason    = cbReadWriteSynch;
    data_s.cb_rtn    = tick_cb;
    data_s.obj       = NULL;
    data_s.time      = &time_s;
    data_s.value     = NULL;
    data_s.user_data = NULL;
    vpi_free_object(vpi_register_cb(&data_s));
  }

  virtual void dumpon() {
    s_vpi_value value_s;
    value_s.format    = vpiHexStrVal;
    value_s.value.str = (PLI_BYTE8*) "1";
    vpi_put_value(dump_handle, &value_s, NULL, vpiNoDelay);
  }

  virtual void dumpoff() {
    s_vpi_value value_s;
    value_s.format    = vpiHexStrVal;
    value_s.value.str = (PLI_BYTE8*) "0";
    vpi_put_value(dump_handle, &value_s, NULL, vpiNoDelay);
  }

  virtual size_t add_signal(vpiHandle& sig_handle, std::string& wire) {
    size_t id = sim_data.signals.size();
    sim_data.signals.push_back(sig_handle);
    sim_data.signal_map[wire] = id;
    return id;
  }

  virtual int search(std::string& wire) {
    std::string topPath = vpi_get_str(vpiFullName, top_handle);
    int dotpos = topPath.find(".");
    std::string path = topPath.substr(0, dotpos) + wire;
    vpiHandle handle = vpi_handle_by_name((PLI_BYTE8 *) path.c_str(), NULL);
    return handle ? add_signal(handle, wire) : -1;
  }

  void probe_signals() {
    std::queue<vpiHandle> modules;
    size_t offset = std::string(vpi_get_str(vpiFullName, top_handle)).find(".") + 1;

    // Start from the top module
    modules.push(top_handle);

    while (!modules.empty()) {
      vpiHandle mod_handle = modules.front();
      modules.pop();

      std::string modname = std::string(vpi_get_str(vpiFullName, mod_handle)).substr(offset);
      // Iterate its nets
      vpiHandle net_iter = vpi_iterate(vpiNet, mod_handle);
      while (vpiHandle net_handle = vpi_scan(net_iter)) {
        std::string netname = vpi_get_str(vpiName, net_handle);
        std::string netpath = modname + "." + netname;
        add_signal(net_handle, netpath);
      }

      // Iterate its regs
      vpiHandle reg_iter = vpi_iterate(vpiReg, mod_handle);
      while (vpiHandle reg_handle = vpi_scan(reg_iter)) {
        std::string regname = vpi_get_str(vpiName, reg_handle);
        std::string regpath = modname + "." + regname;
        add_signal(reg_handle, regpath);
      }

      // Iterate its mems
      vpiHandle mem_iter = vpi_iterate(vpiRegArray, mod_handle);
      while (vpiHandle mem_handle = vpi_scan(mem_iter)) {
        vpiHandle elm_iter = vpi_iterate(vpiReg, mem_handle);
        while (vpiHandle elm_handle = vpi_scan(elm_iter)) {
          std::string elmname = vpi_get_str(vpiName, elm_handle);
          std::string elmpath = modname + "." + elmname;
          add_signal(elm_handle, elmpath);
        }
      }

      // Find DFF
      vpiHandle udp_iter = vpi_iterate(vpiPrimitive, mod_handle);
      while (vpiHandle udp_handle = vpi_scan(udp_iter)) {
        if (vpi_get(vpiPrimType, udp_handle) == vpiSeqPrim) {
          add_signal(udp_handle, modname);
        }
      }

      vpiHandle sub_iter = vpi_iterate(vpiModule, mod_handle);
      while (vpiHandle sub_handle = vpi_scan(sub_iter)) {
        modules.push(sub_handle);
      }
    }
  }
};

#endif // __VPI_H
