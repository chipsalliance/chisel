#ifndef __SIM_API_H
#define __SIM_API_H

#include <cassert>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <queue>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

enum SIM_CMD { RESET, STEP, UPDATE, POKE, PEEK, FORCE, GETID, GETCHK, FIN };
const int SIM_CMD_MAX_BYTES = 1024;
const int channel_data_offset_64bw = 4;	// Offset from start of channel buffer to actual user data in 64bit words.
static size_t gSystemPageSize;

template<class T> struct sim_data_t {
  std::vector<T> resets;
  std::vector<T> inputs;
  std::vector<T> outputs;
  std::vector<T> signals;
  std::map<std::string, size_t> signal_map;
  // Calculate the size (in bytes) of data stored in a vector.
  size_t storage_size(const std::vector<T> vec) {
    int nitems = vec.size();
#ifdef VPI_USER_H
    return nitems * sizeof(T);
#else
    size_t result = 0;
    for (int i = 0; i < nitems; i++) {
      result += vec[i]->get_num_words();
    }
    return result * sizeof(uint64_t);
#endif
  }
};

class channel_t {
public:
#define ROUND_UP(N, S) ((((N) + (S) -1 ) & (~((S) - 1))))
  void init_map() {
	static std::string m_prefix("channel_t::init_map - ");
	// ensure the data is available (a full page worth).
    if (lseek(fd, map_size-1, SEEK_SET) == -1) {
    	perror((m_prefix + "file: " + full_file_path + " seek to end of page").c_str());
    	exit(1);
    }
    if (write(fd, "", 1) == -1) {
    	perror((m_prefix + "file: " + full_file_path + " write byte").c_str());
    	exit(1);
    }
    if (fsync(fd) == -1) {
    	perror((m_prefix + "file: " + full_file_path + " fsync").c_str());
    	exit(1);
    }
    channel = (char*)mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (channel == MAP_FAILED) {
    	perror((m_prefix + "file: " + full_file_path + " mmap").c_str());
    	exit(1);
    }
  }
  channel_t(std::string _file_name, size_t _data_size): file_name(_file_name), fd(open(file_name.c_str(),  O_RDWR|O_CREAT|O_TRUNC, (mode_t)0600)),
		  map_size(ROUND_UP(_data_size + channel_data_offset_64bw * 8, gSystemPageSize)) {
	static std::string m_prefix("channel_t::channel_t: ");
    char * rp = realpath(file_name.c_str(), NULL);
    full_file_path = std::string(rp == NULL ? file_name : rp);
    if (rp != NULL) {
    	free(rp);
    	rp = NULL;
    }
	if (fd == -1) {
    	perror((m_prefix + "file: " + full_file_path + " open").c_str());
		exit(1);
	}
	init_map();
  }

  ~channel_t() {
    munmap((void *)channel, map_size);
    close(fd);
  }
  inline void aquire() {
    channel[1] = 1;
    channel[2] = 1;
    while (channel[0] == 1 && channel[2] == 1);
  }
  inline void release() { channel[1] = 0; }
  inline void produce() { channel[3] = 1; }
  inline void consume() { channel[3] = 0; }
  inline bool ready() { return channel[3] == 0; }
  inline bool valid() { return channel[3] == 1; }
  inline uint64_t* data() { return (uint64_t*)(channel + channel_data_offset_64bw); }
  inline char* str() { return ((char *)channel + channel_data_offset_64bw); }
  inline uint64_t& operator[](int i) { return data()[i*sizeof(uint64_t)]; }
private:
  // Dekker's alg for sync
  // channel[0] -> tester
  // channel[1] -> simulator
  // channel[2] -> turn
  // channel[3] -> flag
  // channel[4:] -> data
  char volatile * channel;
  const std::string file_name;
  std::string full_file_path;
  const int fd;
  const size_t map_size;
};

template <class T> class sim_api_t {
public:
  sim_api_t() {
    // This is horrible, but we'd rather not have to generate another .cpp initialization file,
    //  and have all our clients update their Makefiles (if they don't use ours) to build the simulator.
    if (gSystemPageSize == 0) {
      gSystemPageSize = sysconf(_SC_PAGESIZE);
    }
  }
  void init_channels() {
    pid_t pid = getpid();
    std::ostringstream in_ch_name, out_ch_name, cmd_ch_name;
    in_ch_name  << std::dec << std::setw(8) << std::setfill('0') << pid << ".in";
    out_ch_name << std::dec << std::setw(8) << std::setfill('0') << pid << ".out";
    cmd_ch_name << std::dec << std::setw(8) << std::setfill('0') << pid << ".cmd";
    size_t input_size = this->sim_data.storage_size(this->sim_data.inputs);
    in_channel  = new channel_t(in_ch_name.str(), input_size);
    size_t output_size = this->sim_data.storage_size(this->sim_data.outputs);
    out_channel = new channel_t(out_ch_name.str(), output_size);
    cmd_channel = new channel_t(cmd_ch_name.str(), SIM_CMD_MAX_BYTES);
    
    // Init channels
    out_channel->consume();
    in_channel->release();
    out_channel->release();
    cmd_channel->release();
    // Inform the tester that the simulation is ready
    char hostName[256];
    const char *hostNamep = NULL;
    if (gethostname(hostName, sizeof(hostName) - 1) == 0) {
      hostNamep = hostName;
    } else {
      hostNamep = "<unknown>";
    }
    time_t now;
    time(&now);
    // NOTE: ctime() generates a trailing '\n'.
    std::cerr << "sim start on " << hostNamep << " at " << ctime(&now);
    std::cerr << in_ch_name.str() << std::endl;
    std::cerr << out_ch_name.str() << std::endl;
    std::cerr << cmd_ch_name.str() << std::endl;
  }
  virtual ~sim_api_t() {
    delete in_channel;
    delete out_channel;
    delete cmd_channel;
  }
  virtual void tick() {
    static bool is_reset;
    // First, Send output tokens
    while(!send_tokens());
    if (is_reset) start();
    is_reset = false;

    // Next, handle commands from the testers
    bool is_exit = false;
    do {
      size_t cmd;
      while(!recv_cmd(cmd));
      switch ((SIM_CMD) cmd) {
        case RESET: reset(); is_reset = true; is_exit = true; break;
        case STEP: while(!recv_tokens()); step(); is_exit = true; break;
        case UPDATE: while(!recv_tokens()); update(); is_exit = true; break;
        case POKE: poke(); break; 
        case PEEK: peek(); break;
        case FORCE: poke(true); break;
        case GETID: getid(); break;
        case GETCHK: getchk(); break;
        case FIN: finish(); is_exit = true; break;
        default: break;
      }
    } while (!is_exit);
  }

protected:
  sim_data_t<T> sim_data;

private:
  channel_t *in_channel;
  channel_t *out_channel;
  channel_t *cmd_channel;

  virtual void reset() = 0;
  virtual void start() = 0; 
  virtual void finish() = 0;
  virtual void step() = 0;
  virtual void update() = 0; 
  // Consumes input tokens 
  virtual size_t put_value(T& sig, uint64_t* data, bool force = false) = 0;
  // Generate output tokens
  virtual size_t get_value(T& sig, uint64_t* data) = 0;
  // Find a signal of path 
  virtual int search(std::string& path) { return -1; }
  virtual size_t get_chunk(T& sig) = 0;

  void poke(bool force = false) {
    size_t id;
    while(!recv_cmd(id));
    T obj = sim_data.signals[id];
    if (!obj) {
      std::cerr << "Cannot find the object of id = " << id << std::endl;
      finish();
      exit(2);		// Not a normal exit.
    }
    while(!recv_value(obj, force));
  }

  void peek() {
    size_t id;
    while(!recv_cmd(id));
    T obj = sim_data.signals[id];
    if (!obj) {
      std::cerr << "Cannot find the object of id = " << id << std::endl;
      finish();
      exit(2);		// Not a normal exit.
    }
    while(!send_value(obj));
  }

  void getid() {
    std::string path;
    while(!recv_cmd(path));
    std::map<std::string, size_t>::iterator it = sim_data.signal_map.find(path);
    if (it != sim_data.signal_map.end()) {
      while(!send_resp(it->second));
    } else {
      int id = search(path);
      if (id < 0) {
    	// Issue warning message but don't exit here.
        std::cerr << "Cannot find the object, " << path << std::endl;
      }
      while(!send_resp(id));
    }
  }

  void getchk() {
    size_t id;
    while(!recv_cmd(id));
    T obj = sim_data.signals[id];
    if (!obj) {
      std::cerr << "Cannot find the object of id = " << id << std::endl;
      finish();
      exit(2);		// Not a normal exit.
    }
    size_t chunk = get_chunk(obj);
    while(!send_resp(chunk));
  }

  bool recv_cmd(size_t& cmd) {
    cmd_channel->aquire();
    bool valid = cmd_channel->valid();
    if (valid) {
      cmd = (*cmd_channel)[0];
      cmd_channel->consume();
    }
    cmd_channel->release();
    return valid;
  }

  bool recv_cmd(std::string& path) {
    cmd_channel->aquire();
    bool valid = cmd_channel->valid();
    if (valid) {
      path = cmd_channel->str();
      cmd_channel->consume();
    }
    cmd_channel->release();
    return valid;
  }

  bool send_resp(size_t value) {
    out_channel->aquire();
    bool ready = out_channel->ready();
    if (ready) {
      (*out_channel)[0] = value;
      out_channel->produce();
    }
    out_channel->release();
    return ready;
  }

  bool recv_value(T& obj, bool force = false) {
    in_channel->aquire();
    bool valid = in_channel->valid();
    if (valid) {
      put_value(obj, in_channel->data(), force);
      in_channel->consume();
    }
    in_channel->release();
    return valid;
  }

  bool send_value(T& obj) {
    out_channel->aquire();
    bool ready = out_channel->ready();
    if (ready) {
      get_value(obj, out_channel->data());
      out_channel->produce();
    }
    out_channel->release();
    return ready;
  }

  bool recv_tokens() {
    in_channel->aquire();
    bool valid = in_channel->valid();
    if (valid) {
      size_t off = 0;
      uint64_t *data = in_channel->data();
      for (size_t i = 0 ; i < sim_data.inputs.size() ; i++) {
        T& sig = sim_data.inputs[i];
        off += put_value(sig, data+off);
      }
      in_channel->consume();
    }
    in_channel->release();
    return valid;
  }

  bool send_tokens() {
    out_channel->aquire();
    bool ready = out_channel->ready();
    if (ready) {
      size_t off = 0;
      uint64_t *data = out_channel->data();
      for (size_t i = 0 ; i < sim_data.outputs.size() ; i++) {
        T& sig = sim_data.outputs[i];
        off += get_value(sig, data+off);
      }
      out_channel->produce();
    }
    out_channel->release();
    return ready;
  }
};

#endif //__SIM_API_H
