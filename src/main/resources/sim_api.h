#ifndef __SIM_API_H
#define __SIM_API_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

enum SIM_CMD { RESET, STEP, UPDATE, POKE, PEEK, GETID, SETCLK, FIN };

template<class T> struct sim_data_t {
  std::vector<T> resets;
  std::vector<T> inputs;
  std::vector<T> outputs;
  std::vector<T> signals;
  std::map<std::string, size_t> signal_map;
  std::map<std::string, T> clk_map;
};

template <class T> class sim_api_t {
public:
  void tick() {
    static bool is_reset = false;
    // First, Generates output tokens  (in hex)
    generate_tokens();
    if (is_reset) {
      start();
      is_reset = false;
    }
    
    // Next, handle commands from the testers
    bool exit = false;
    do {
      size_t cmd;
      std::cin >> std::dec >> cmd;
      switch ((SIM_CMD) cmd) {
        case RESET: 
          reset(); is_reset = true; exit = true; break;
        case STEP: 
          consume_tokens();
          step(); exit = true; break;
        case UPDATE: 
          consume_tokens();
          update(); exit = true; break;
        case POKE: poke(); break; 
        case PEEK: peek(); break;
        case GETID: getid(); break;
        case SETCLK: setclk(); break;
        case FIN:  finish(); exit = true; break;
        default: break;
      }
    } while (!exit);
  }
private:
  virtual void reset() = 0;
  virtual void start() = 0; 
  virtual void finish() = 0;
  virtual void update() = 0; 
  virtual void step() = 0;
  // Consumes input tokens (in hex)
  virtual void put_value(T& sig) = 0;
  // Generate output tokens (in hex)
  virtual void get_value(T& sig) = 0;
  // Find a signal of path 
  virtual int search(std::string& path) { return -1; }

  void poke() {
    size_t id;
    std::cin >> std::dec >> id;
    T obj = sim_data.signals[id];
    if (obj) {
      put_value(obj);
    } else {
      std::cout << "Cannot find the object of id = " << id << std::endl;
      finish();
      exit(0);
    }
  }

  void peek() {
    size_t id;
    std::cin >> std::dec >> id;
    T obj = sim_data.signals[id];
    if (obj) {
      get_value(obj);
    } else {
      std::cout << "Cannot find the object of id = " << id << std::endl;
      finish();
      exit(0);
    }
  }

  void getid() {
    std::string wire;
    std::cin >> wire;
    std::map<std::string, size_t>::iterator it = sim_data.signal_map.find(wire);
    if (it != sim_data.signal_map.end()) {
      std::cerr << it->second << std::endl;
    } else {
      int id = search(wire);
      if (id < 0) {
        std::cout << "Cannot find the object, " << wire<< std::endl;
        finish();
        exit(0);
      }
      std::cerr << id << std::endl;
    }
  }

  void setclk() {
    std::string clkname;
    std::cin >> clkname;
    typename std::map<std::string, T>::iterator it = sim_data.clk_map.find(clkname);
    if (it != sim_data.clk_map.end()) {
      put_value(it->second);
    } else {
      std::cout << "Cannot find " << clkname << std::endl;
    }
  }

  void consume_tokens() {
    for (size_t i = 0 ; i < sim_data.inputs.size() ; i++) {
      put_value(sim_data.inputs[i]);
    }
  }

  void generate_tokens() {
    for (size_t i = 0 ; i < sim_data.outputs.size() ; i++) {
      get_value(sim_data.outputs[i]);
    }
  }
protected:
  sim_data_t<T> sim_data;

  void read_signal_map(std::string filename) {
    std::ifstream file(filename.c_str());
    if (!file) {
      std::cout << "Cannot open " << filename << std::endl;
      finish();
      exit(0);
    } 
    std::string line;
    size_t id = 0;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string path;
      size_t width, n;
      iss >> path >> width >> n;
      sim_data.signal_map[path] = id;
      id += n;
    }
  }
};

#endif //__SIM_API_H
