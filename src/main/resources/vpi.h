#ifndef __VPI_H
#define __VPI_h

#include "vpi_user.h"
#include "sim_api.h"
#include <queue>

PLI_INT32 tick_cb(p_cb_data cb_data);

class vpi_api_t: public sim_api_t<vpiHandle> {
public:
  void init_clks() {
    vpiHandle syscall_handle = vpi_handle(vpiSysTfCall, NULL);
    vpiHandle arg_iter = vpi_iterate(vpiArgument, syscall_handle);
    // Cache clocks
    while (vpiHandle arg_handle = vpi_scan(arg_iter)) {
      std::string name = vpi_get_str(vpiName, arg_handle);
      sim_data.clk_map[name.substr(0, name.rfind("_len"))] = arg_handle;
    }
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
    top_handle = vpi_scan(vpi_iterate(vpiArgument, syscall_handle));
    search_signals();
  }

private:
  vpiHandle top_handle;

  void put_value(vpiHandle& sig) {
    std::string value;
    for (size_t k = 0 ; k < ((vpi_get(vpiSize, sig) - 1) >> 6) + 1 ; k++) {
      // 64 bit chunks are given
      std::string v;
      std::cin >> v;
      value += v;
    }
    s_vpi_value value_s;
    value_s.format = vpiHexStrVal;
    value_s.value.str = (PLI_BYTE8*) value.c_str();
    vpi_put_value(sig, &value_s, NULL, vpiNoDelay);
  }

  void get_value(vpiHandle& sig) {
    s_vpi_value value_s;
    value_s.format = vpiHexStrVal;
    vpi_get_value(sig, &value_s);
    std::cerr << value_s.value.str << std::endl;
  }

  virtual void reset() {
    for (size_t i = 0 ; i < sim_data.resets.size() ; i++) {
      s_vpi_value value_s;
      value_s.format = vpiHexStrVal;
      value_s.value.str = (PLI_BYTE8*) "1";
      vpi_put_value(sim_data.resets[i], &value_s, NULL, vpiNoDelay);
    }
  }

  virtual void start() {
    for (size_t i = 0 ; i < sim_data.resets.size() ; i++) {
      s_vpi_value value_s;
      value_s.format = vpiHexStrVal;
      value_s.value.str = (PLI_BYTE8*) "0";
      vpi_put_value(sim_data.resets[i], &value_s, NULL, vpiNoDelay);
    }
  }

  virtual void finish() { vpi_control(vpiFinish, 0); }

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

  virtual size_t add_signal(vpiHandle& sig_handle, std::string& wire) {
    size_t id = sim_data.signals.size();
    sim_data.signals.push_back(sig_handle);
    sim_data.signal_map[wire] = id;
    return id;
  }

  int search_signals(const char *wire = NULL) {
    int id = -1;
    std::string wirepath = wire ? wire : "";
    int dotpos = wirepath.rfind(".");
    std::string modpath = dotpos > 0 ? wirepath.substr(0, dotpos) : "";
    std::string wirename = dotpos > 0 ? wirepath.substr(dotpos+1) : "";
    int sbrpos = wirename.rfind("[");
    std::string arrname = sbrpos > 0 ? wirename.substr(0, sbrpos) : "";
    std::queue<vpiHandle> modules;
    size_t offset = std::string(vpi_get_str(vpiFullName, top_handle)).find(".") + 1;

    // Start from the top module
    modules.push(top_handle);

    while (!modules.empty()) {
      vpiHandle mod_handle = modules.front();
      modules.pop();

      std::string modname = std::string(vpi_get_str(vpiFullName, mod_handle)).substr(offset);
      // If the module is found
      if (!wire || modpath == modname) {
        // Iterate its nets
        vpiHandle net_iter = vpi_iterate(vpiNet, mod_handle);
        while (vpiHandle net_handle = vpi_scan(net_iter)) {
          std::string netname = vpi_get_str(vpiName, net_handle);
          std::string netpath = modname + "." + netname;
          size_t netid = (!wire && netname[0] != 'T') || wirename == netname ? 
            add_signal(net_handle, netpath) : 0;
          id = netid ? netid : id;
          if (id > 0) break;
        }
        if (id > 0) break;

        // Iterate its regs
        vpiHandle reg_iter = vpi_iterate(vpiReg, mod_handle);
        while (vpiHandle reg_handle = vpi_scan(reg_iter)) {
          std::string regname = vpi_get_str(vpiName, reg_handle);
          std::string regpath = modname + "." + regname;
          size_t regid = !wire || wirename == regname ? 
            add_signal(reg_handle, regpath) : 0;
          id = regid ? regid : id;
          if (id > 0) break;
        }
        if (id > 0) break;

        // Iterate its mems
        vpiHandle mem_iter = vpi_iterate(vpiRegArray, mod_handle);
        while (vpiHandle mem_handle = vpi_scan(mem_iter)) {
          std::string memname = vpi_get_str(vpiName, mem_handle);
          if (!wire || arrname == memname) {
            vpiHandle elm_iter = vpi_iterate(vpiReg, mem_handle);
            size_t idx = vpi_get(vpiSize, mem_handle);
            while (vpiHandle elm_handle = vpi_scan(elm_iter)) {
              std::string elmname = vpi_get_str(vpiName, elm_handle);
              std::string elmpath = modname + "." + elmname;
              size_t elmid = add_signal(elm_handle, elmpath);
              id = wirename == elmname ? elmid : id;
            }
          }
          if (id > 0) break;
        }
      }

      // Find DFF
      if (!wire || wirepath == modname) {
        vpiHandle udp_iter = vpi_iterate(vpiPrimitive, mod_handle);
        while (vpiHandle udp_handle = vpi_scan(udp_iter)) {
          if (vpi_get(vpiPrimType, udp_handle) == vpiSeqPrim) {
            id = add_signal(udp_handle, modname);
            break;
          }
        }
      }
      if (id > 0) break;

      vpiHandle sub_iter = vpi_iterate(vpiModule, mod_handle);
      while (vpiHandle sub_handle = vpi_scan(sub_iter)) {
        modules.push(sub_handle);
      }
    }

    return id;
  }

  virtual int search(std::string& wire) {
    return search_signals(wire.c_str());
  }
};

#endif // __VPI_H
