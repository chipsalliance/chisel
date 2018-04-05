#ifndef __SIM_API_H
#define __SIM_API_H

#ifdef __WINNT
#include <windows.h>
#endif
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
#ifndef __WINNT
#include <sys/mman.h>
#endif
#include <time.h>

extern "C"
{
  // From local.
  int fsync (int fd);
}

enum SIM_CMD { RESET, STEP, UPDATE, POKE, PEEK, FORCE, GETID, GETCHK, FIN };
const int SIM_CMD_MAX_BYTES = 1024;
const int channel_data_offset_64bw = 4;	// Offset from start of channel buffer to actual user data in 64bit words.
static size_t gSystemPageSize;
typedef std::string sim_data_map_key;

template<class T> struct sim_data_t {
  std::vector<T> resets;
  std::vector<T> inputs;
  std::vector<T> outputs;
  std::vector<T> signals;
  std::map<sim_data_map_key, size_t> signal_map;
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

class ChannelProtocol {
public:
    void setBuffer(volatile char *a_channel) {
        channel = a_channel;
        base = (void *)channel;
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
    void* base;
private:
    // Dekker's alg for sync
    // channel[0] -> tester
    // channel[1] -> simulator
    // channel[2] -> turn
    // channel[3] -> flag
    // channel[4:] -> data
    char volatile * channel;
};

class NullChannel: public ChannelProtocol {
public:
    inline void aquire() {}
    inline void release() {}
    inline void produce() {}
    inline void consume() {}
    inline bool ready() { true; }
    inline bool valid() { false; }
    inline uint64_t* data() { return NULL; }
    inline char* str() { return NULL; }
    inline uint64_t& operator[](int i) { return dummy; }
private:
    uint64_t dummy;
};

class SharedMap {
public:
    volatile char * init_map() {
        static std::string m_prefix("sim_api.init_map - ");
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
#ifdef __WINNT
        hMapFile = CreateFileMapping((HANDLE)_get_osfhandle(fd), NULL, PAGE_READWRITE, 0, 0, NULL);
        if (hMapFile == INVALID_HANDLE_VALUE) {
            DWORD errorVal = GetLastError();
            LPTSTR lpBuffer;
            DWORD fmResult = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM,
                NULL,
                errorVal,
                0,
                (LPTSTR)&lpBuffer,
                80,
                NULL
					   );
            std::string errorMessage = m_prefix + "file: " + full_file_path + " CreateFileMapping: " + "%s (%d)";
            fprintf(stderr, errorMessage.c_str(), lpBuffer, errorVal);
            LocalFree(lpBuffer);
            exit(1);
        }
        volatile char * buffer = (volatile char *)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, map_size);
        if (buffer == NULL) {
            DWORD errorVal = GetLastError();
            LPTSTR lpBuffer;
            DWORD fmResult = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM,
                NULL,
                errorVal,
                0,
                (LPTSTR)&lpBuffer,
                80,
                NULL
					   );
            std::string errorMessage = m_prefix + "file: " + full_file_path + " MapViewOfFile: " + "%s (%d)";
            fprintf(stderr, errorMessage.c_str(), lpBuffer, errorVal);
            LocalFree(lpBuffer);
            exit(1);

        }
#else
        volatile char * buffer = (volatile char *)mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (buffer == MAP_FAILED) {
            perror((m_prefix + "file: " + full_file_path + " mmap").c_str());
            exit(1);
        }
#endif
        return buffer;
    }

#define ROUND_UP(N, S) ((((N) + (S) -1 ) & (~((S) - 1))))
    SharedMap(std::string _file_name, size_t _data_size): file_name(_file_name),
#ifdef __WINNT
    hMapFile(INVALID_HANDLE_VALUE),
#endif
    fd(open(file_name.c_str(),  O_RDWR|O_CREAT|O_TRUNC, (mode_t)0600)),
    map_size(ROUND_UP(_data_size + channel_data_offset_64bw * 8, gSystemPageSize))
        {
        static std::string m_prefix("channel_t::channel_t: ");
#ifdef __WINNT
	char *rp = NULL;
#else
        char * rp = realpath(file_name.c_str(), NULL);
#endif
        full_file_path = std::string(rp == NULL ? file_name : rp);
        if (rp != NULL) {
            free(rp);
            rp = NULL;
        }
        if (fd == -1) {
            perror((m_prefix + "file: " + full_file_path + " open").c_str());
            exit(1);
        }
        mapped_buffer = init_map();
    }
  ~SharedMap() {
#ifdef __WINNT
    if (mapped_buffer) {
      UnmapViewOfFile((LPCVOID)mapped_buffer);
        mapped_buffer = NULL;
    }
    CloseHandle(hMapFile);
    hMapFile = INVALID_HANDLE_VALUE;
#else
    munmap((void *)mapped_buffer, map_size);
#endif
    close(fd);
  }
  volatile char * mapped_buffer;
  const std::string file_name;
  std::string full_file_path;
#ifdef __WINNT
  HANDLE hMapFile;
#endif
  const int fd;
  const size_t map_size;
};

class channel_t: public ChannelProtocol {
public:
    channel_t(volatile char * mapped_buffer) {
      setBuffer(mapped_buffer);
    }
};

class SIM_API {
public:
  virtual void reset() = 0;
  virtual void start() = 0;
  virtual void finish() = 0;
  virtual void step() = 0;
  virtual void update() = 0;
  // Find a signal of path
  virtual int search(sim_data_map_key path) = 0;
  virtual int getSignalWordSize(int id) = 0;
  virtual void peekn(int id, int n, uint64_t *data) = 0;
  virtual void poken(int id, int n, uint64_t *data) = 0;
  virtual void setInputSignals(uint64_t *data) = 0;
  virtual void getOutputSignals(uint64_t *data) = 0;
  virtual void init_channels(volatile char * inputBuffer, volatile char * outputBuffer, volatile char * cmdBuffer) = 0;
  virtual void getInputChannelBuffer(void *& address, long& size) = 0;
  virtual void getOutputChannelBuffer(void *& address, long& size) = 0;
  virtual bool send_tokens() = 0;
  virtual bool recv_tokens() = 0;
};

struct ChannelBuffers {
    const void * inputBuffer;
    const void * outputBuffer;
    const void * cmdBuffer;
private:
    void markBuffer(char * cp, int size) {
        for (int i = 0; i < size; i += 1)
            cp[i] = 1;
    }
public:
    ChannelBuffers(size_t input_size, size_t output_size, size_t cmd_size):
        inputBuffer(input_size ? malloc(input_size) : (void *)0),
        outputBuffer(output_size ? malloc(output_size) : (void *)0),
        cmdBuffer(cmd_size ? malloc(cmd_size) : (void *)0)
    {
        markBuffer((char *)inputBuffer, input_size);
        markBuffer((char *)outputBuffer, output_size);
        markBuffer((char *)cmdBuffer, cmd_size);
//        std::cerr << "sim_api_t::ChannelBuffers: inputBuffer " << inputBuffer << ", size " << input_size << std::endl;
//        std::cerr << "sim_api_t::ChannelBuffers: outputBuffer " << outputBuffer << ", size " << output_size << std::endl;
//        std::cerr << "sim_api_t::ChannelBuffers: cmdBuffer " << cmdBuffer << ", size " << cmd_size << std::endl;
    }
    ~ChannelBuffers() {
        free((void *)inputBuffer);
        free((void *)outputBuffer);
        free((void *)cmdBuffer);
    }
};

class IPCChannels {
public:
    IPCChannels(size_t input_size, size_t output_size) {
        // This is horrible, but we'd rather not have to generate another .cpp initialization file,
        //  and have all our clients update their Makefiles (if they don't use ours) to build the simulator.
        if (gSystemPageSize == 0) {
#ifdef __WINNT
          SYSTEM_INFO systemInfo;
          GetNativeSystemInfo(&systemInfo);
          gSystemPageSize = systemInfo.dwPageSize;

#else
          gSystemPageSize = sysconf(_SC_PAGESIZE);
#endif
        }
        pid_t pid = getpid();
        in_ch_name  << std::dec << std::setw(8) << std::setfill('0') << pid << ".in";
        out_ch_name << std::dec << std::setw(8) << std::setfill('0') << pid << ".out";
        cmd_ch_name << std::dec << std::setw(8) << std::setfill('0') << pid << ".cmd";
        in_sharedMap  = new SharedMap(in_ch_name.str(), input_size);
        out_sharedMap = new SharedMap(out_ch_name.str(), output_size);
        cmd_sharedMap = new SharedMap(cmd_ch_name.str(), SIM_CMD_MAX_BYTES);
    }
    void announce() {
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
    virtual ~IPCChannels() {
        delete in_sharedMap;
        delete out_sharedMap;
        delete cmd_sharedMap;
    }
    SharedMap *in_sharedMap;
    SharedMap *out_sharedMap;
    SharedMap *cmd_sharedMap;
private:
    std::ostringstream in_ch_name, out_ch_name, cmd_ch_name;
};

template <class T> class sim_api_t: public SIM_API {

public:
    sim_api_t(): channelBuffers(0) {
    }
    void init_channels(ChannelBuffers *bp) {
        in_channel  = bp->inputBuffer ? (ChannelProtocol *)new channel_t((volatile char*)bp->inputBuffer) : new NullChannel();
        out_channel = bp->outputBuffer ? (ChannelProtocol *)new channel_t((volatile char*)bp->outputBuffer) : new NullChannel();
        cmd_channel = bp->cmdBuffer ? (ChannelProtocol *)new channel_t((volatile char*)bp->cmdBuffer) : new NullChannel();
        channelBuffers = bp;
    }
    void init_channels(volatile char * inputBuffer, volatile char * outputBuffer, volatile char * cmdBuffer) {
//        std::cerr << std::hex << "sim_api_t::init_channels: in 0x" << inputBuffer << ", out 0x" << outputBuffer << ", cmd 0x" << cmdBuffer << std::endl;
//        fprintf(stderr, "sim_api_t::init_channels: in 0x%lx, out 0x%lx, cmd 0x%lx\n", inputBuffer, outputBuffer, cmdBuffer);
        in_channel  = inputBuffer ? (ChannelProtocol *)new channel_t(inputBuffer) : new NullChannel();
        out_channel = outputBuffer ? (ChannelProtocol *)new channel_t(outputBuffer) : new NullChannel();
        cmd_channel = cmdBuffer ? (ChannelProtocol *)new channel_t(cmdBuffer) : new NullChannel();

        // Init channels
//        std::cerr << "sim_api_t::init_channels: out consume" << std::endl;
        out_channel->consume();
//        std::cerr << "sim_api_t::init_channels: in release" << std::endl;
        in_channel->release();
//        std::cerr << "sim_api_t::init_channels: out release" << std::endl;
        out_channel->release();
//        std::cerr << "sim_api_t::init_channels: cmd release" << std::endl;
        cmd_channel->release();
    }
  virtual ~sim_api_t() {
    delete in_channel;
    delete out_channel;
    delete cmd_channel;
    delete channelBuffers;
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

    size_t getInputSize() {
     size_t result = this->sim_data.storage_size(this->sim_data.inputs) + channel_data_offset_64bw;
//     fprintf(stderr, "sim_api_t::getInputSize: %d\n", result);
     return result;
    }
    size_t getOutputSize() {
     size_t result = this->sim_data.storage_size(this->sim_data.outputs) + channel_data_offset_64bw;
//     fprintf(stderr, "sim_api_t::getOutputSize: %d\n", result);
     return result;
    }

protected:
  sim_data_t<T> sim_data;

private:
  ChannelProtocol *in_channel;
  ChannelProtocol *out_channel;
  ChannelProtocol *cmd_channel;
  ChannelBuffers *channelBuffers;

  // Consumes input tokens
  virtual size_t put_value(T& sig, uint64_t* data, bool force = false) = 0;
  // Generate output tokens
  virtual size_t get_value(T& sig, uint64_t* data) = 0;
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

  void poken(int id, int n, uint64_t * data) {
    int m = getSignalWordSize(id);
    if (n != m) {
//      std::cerr << "poken(" << id << ", " << n << ", ...) " << n << " != " << m << std::endl;
    }
    T obj = sim_data.signals[id];
    bool force = false;
    if (obj) {
      put_value(obj, data, force);
    }
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

  void peekn(int id, int n, uint64_t * data) {
    int m = getSignalWordSize(id);
    if (n != m) {
//      std::cerr << "peekn(" << id << ", " << n << ", ...) " << n << " != " << m << std::endl;
    } else {
     T obj = sim_data.signals[id];
     if (obj) {
       get_value(obj, data);
     }
   }
  }

  void getid() {
    std::string path;
    while(!recv_cmd(path));
    sim_data_map_key signalName = path;
    std::map<sim_data_map_key, size_t>::iterator it = sim_data.signal_map.find(path);
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

  int getSignalWordSize(int id) {
    int chunk = -1;
    T obj = sim_data.signals[id];
    if (obj) {
        chunk = get_chunk(obj);
    }
    return chunk;
  }

  void getchk() {
    size_t id;
    while(!recv_cmd(id));
    int chunk = getSignalWordSize(id);
    if (chunk < 0) {
      std::cerr << "Cannot find the object of id = " << id << std::endl;
      finish();
      exit(2);		// Not a normal exit.
    }
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

  void getInputChannelBuffer(void *& address, long& size) {
      address = in_channel->base;
      size = getInputSize();
//      fprintf(stderr, "api_sim_t::getInputChannelBuffer: address = 0x%lx, size %ld", address, size);
  }

  void setInputSignals(uint64_t *data) {
      size_t off = 0;
      for (size_t i = 0 ; i < sim_data.inputs.size() ; i++) {
        T& sig = sim_data.inputs[i];
        off += put_value(sig, data+off);
      }
  }

  bool recv_tokens() {
    in_channel->aquire();
    bool valid = in_channel->valid();
    if (valid) {
      setInputSignals(in_channel->data());
      in_channel->consume();
    }
    in_channel->release();
    return valid;
  }

  void getOutputChannelBuffer(void *& address, long& size) {
      address = out_channel->base;
      size = getOutputSize();
//      fprintf(stderr, "api_sim_t::getOutputChannelBuffer: address = 0x%lx, size %ld", address, size);
  }

  void getOutputSignals(uint64_t * data) {
      size_t off = 0;
      for (size_t i = 0 ; i < sim_data.outputs.size() ; i++) {
        T& sig = sim_data.outputs[i];
        size_t nw = get_value(sig, data+off);
        uint64_t val = data[off];
        off += nw;
//        fprintf(stderr, "api_sim_t::getOutputSignals: %d - size %d, value 0x%lx\n", i, nw, val);
      }
  }

  bool send_tokens() {
    out_channel->aquire();
    bool ready = out_channel->ready();
    if (ready) {
      getOutputSignals(out_channel->data());
      out_channel->produce();
    }
    out_channel->release();
    return ready;
  }
};

#endif //__SIM_API_H
