// Header for Chisel emulator API
#ifndef __IS_EMULATOR_API__
#define __IS_EMULATOR_API__

#include "emulator_mod.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include <string>
#include <sstream>
#include <map>
#include <cassert>
#include <cerrno>

/**
 * Converts an integer to a std::string without needing additional libraries
 * or C++11.
 */
static std::string itos(int in) {
	std::stringstream out;
	out << in;
	return out.str();
}

/**
 * Copy one val_t array to another.
 * nb must be the exact number of bits the val_t represents.
 */
static __attribute__((unused)) void val_cpy(val_t* dst, val_t* src, int nb) {
    for (int i=0; i<val_n_words(nb); i++) {
        dst[i] = src[i];
    }
}

/**
 * Empty a val_t array (sets to zero).
 * nb must be the exact number of bits the val_t represents.
 */
static void val_empty(val_t* dst, int nb) {
    for (int i=0; i<val_n_words(nb); i++) {
        dst[i] = 0;
    }
}

/**
 * Set a val_t array to a integer number. Obviously, the maximum integer
 * is capped by the width of a single val_t element.
 * nb must be the exact number of bits the val_t represents.
 */
static __attribute__((unused)) void val_set(val_t* dst, val_t nb, val_t num) {
    val_empty(dst, nb);
    dst[0] = num;
}

/**
 * Sets a dat_t from a string, where the input radix is automatically determined
 * from the string (or defaults to 10).
 * Returns true on success.
 */
template <int w>
bool dat_from_str(std::string in, dat_t<w>& res, int pos = 0) {
    int radix = 10;
    int negate = 0;

    /* Handle leading minus sign. */
    if (!in.substr(pos, 1).compare("-")) {
        pos++;
        negate = 1;
    }

    if (!in.substr(pos, 1).compare("d")) {
        radix = 10;
        pos++;
    } else if (!in.substr(pos, 1).compare("h")
               || !in.substr(pos, 1).compare("x")) {
        radix = 16;
        pos++;
    } else if (!in.substr(pos, 2).compare("0h")
               || !in.substr(pos, 2).compare("0x")) {
        radix = 16;
        pos += 2;
    } else if (!in.substr(pos, 1).compare("b")) {
        radix = 2;
        pos++;
    } else if (!in.substr(pos, 2).compare("0b")) {
        radix = 2;
        pos += 2;
    }

    const int log_max_radix = 4;
    assert(radix <= (1 << log_max_radix));

    dat_t<w> curr_base = 1;
    res = 0;

    for (int rpos=in.length()-1; rpos>=pos; rpos--) {
        char c = in[rpos];
        val_t c_val = 0;
        if (c == '_') {
            continue;
        }
        if (c >= '0' && c <= '9') {
            c_val = c - '0';
        } else if (c >= 'a' && c <= 'z') {
            c_val = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'Z') {
            c_val = c - 'A' + 10;
        } else {
            std::cerr << "dat_from_str: Invalid character '" << c << "' in '" << in << "'" << std::endl;
            return false;
        }
        if (c_val > radix /* || c_val < 0 */) {
            std::cerr << "dat_from_str: Invalid character '" << c << "'" <<
            		std::endl;
            return false;
        }

        dat_t<w> temp_prod = curr_base * dat_t<log_max_radix>(c_val);
        res = res + temp_prod;
        curr_base = curr_base * dat_t<log_max_radix+1>(radix);
    }
    if (negate) {
        res = -res;
    }
    return true;
}

// API base class, providing common functions
class api_base {
public:
	api_base(const char* new_name, const char* new_path) :
		name(new_name),
		path(new_path)
	{}
	// returns the fully qualified name of this object (path + dot + name)
	std::string get_pathname() {
		if (*path == '\0') {
			return name;
		} else {
			return get_path() + "." + name;
		}
	}
	// returns the short name of this object
	std::string get_name() {
		return name;
	}
	// returns the path of this object (without a trailing dot)
	std::string get_path() {
		return path;
	}
protected:
	const char* name;
	const char* path;
};

// API base (non width templated) class for API accessors to dat_t
class dat_api_base : public api_base {
public:
	dat_api_base(const char* new_name, const char* new_path) :
		api_base(new_name, new_path)
	{}
	// returns the value of this wire as a string, or empty string on failure
	virtual std::string get_value() = 0;
	// sets the value of this wire from a string, returning true on success
	virtual bool set_value(std::string value) = 0;
	// returns the bitwidth of this wire
	virtual std::string get_width() = 0;
};

// dat_api dummy class, does nothing except for return errors
// to be used when a real dat_api object can't be found
class dat_dummy : public dat_api_base {
public:
	dat_dummy() :
		dat_api_base("error", "")
	{}
	std::string get_value() {
		return "error";
	}

	bool set_value(std::string value) {
		return false;
	}

	std::string get_width() {
		return "error";
	}
};

template<int w> class dat_api : public dat_api_base {
public:
	dat_api(dat_t<w>* new_dat, const char* new_name, const char* new_path) :
		dat_api_base(new_name, new_path),
		dat_ptr(new_dat)
	{}

	std::string get_value() {
		return dat_ptr->to_str();
	}

	bool set_value(std::string value) {
		return dat_from_str<w>(value, *dat_ptr);
	}

	std::string get_width() {
		return itos(w);
	}

protected:
	dat_t<w>* dat_ptr;
};

// API base (non width/depth templated) class for API accessors to mem_t
class mem_api_base : public api_base {
public:
	mem_api_base(const char* new_name, const char* new_path) :
		api_base(new_name, new_path)
	{}
	// return the value of an element as a string, or empty string on failure
	virtual std::string get_element(std::string index) = 0;
	// sets the value of an element from a string, returning true on success
	virtual bool set_element(std::string index, std::string value) = 0;
	// returns the bitwidth of a memory element
	virtual std::string get_width() = 0;
	// returns the number of memory elements
	virtual std::string get_depth() = 0;
};

// mem_api dummy class, does nothing except for return errors
// to be used when a real mem_api object can't be found
class mem_dummy : public mem_api_base {
public:
	mem_dummy() :
		mem_api_base("error", "")
	{}
	string get_element(std::string index) {
		return "error";
	}

	bool set_element(std::string index, std::string value) {
		return false;
	}

	std::string get_width() {
		return "error";
	}

	std::string get_depth() {
		return "error";
	}
};

template<int w, int d> class mem_api : public mem_api_base {
public:
	mem_api(mem_t<w, d>* new_mem, const char* new_name, const char* new_path) :
		mem_api_base(new_name, new_path),
		mem_ptr(new_mem)
	{}

	string get_element(std::string index) {
		int index_int = atoi(index.c_str());
		return mem_ptr->contents[index_int].to_str();
	}

	bool set_element(std::string index, std::string value) {
		int index_int = atoi(index.c_str());
		return dat_from_str<w>(value, mem_ptr->contents[index_int]);
	}

	std::string get_width() {
		return itos(w);
	}

	std::string get_depth() {
		return itos(d);
	}

protected:
	mem_t<w, d>* mem_ptr;
};

class mod_api_t {
public:
	mod_api_t():
		teefile(NULL)
	{}

	void init(mod_t* new_module) {
		module = new_module;
		init_mapping_table();
	}

	void set_teefile(FILE* new_teefile) {
		teefile = new_teefile;
	}

	mod_t* get_module() {
		return module;
	}

	// API basic functions
	std::string get_host_name() {return "C++ Emulator API";}
	std::string get_api_version() {return "0";}
	std::string get_api_support() {return "PeekPoke Introspection";}

	// External access functions & helpers
	std::vector< std::string > tokenize(std::string str) {
	    std::vector< std::string > res;
	    int i = 0;
	    int c = ' ';
	    while ( i < str.size() ) {
	      while (isspace(c)) {
	        if (i >= str.size()) return res;
	        c = str[i++];
	      }
	      std::string s;
	      while (!isspace(c) && i < str.size()) {
	        s.push_back(c);
	        c = str[i++];
	      }
	      if (i >= str.size()) s.push_back(c);
	      if (s.size() > 0)
	        res.push_back(s);
	    }
	    return res;
	}

	// helper to verify command length, returning false and printing an error
	// to stderr if the length isn't in the specified range
	bool check_command_length(std::vector<std::string>& tokenized_command,
			int min_args, int max_args=-1) {
		if (tokenized_command.size() - 1 < min_args) {
			std::cerr << tokenized_command[0] << " expects at least " << min_args
					<< " args, got " << tokenized_command.size() - 1
					<< std::endl;
			return false;
		} else if (max_args >= 0 && tokenized_command.size() - 1 > max_args) {
			std::cerr << tokenized_command[0] << " expects at most " << max_args
					<< " args, got " << tokenized_command.size() - 1
					<< std::endl;
			return false;
		}
		return true;
	}

	// Evaluates an API command, returning the reply as a string (without
	// the trailing newline).
	// Errors return "error", printing a more detailed description to stderr.
	// TODO: find a way to pass errors in-line, so transport layers other than
	// stdin/stdout (like TCP/IP) are possible while also communicating errors.
	std::string eval_command(string command) {
		std::vector<std::string> tokens = tokenize(command);
		if (tokens.size() == 0) {
			std::cerr << "Empty command: '" << command << "'" << std::endl;
			return "error";
		}
		if (tokens[0] == "get_host_name") {
			// IN:  get_host_name
			// OUT: API host's name (arbitrary string)
			if (!check_command_length(tokens, 0, 0)) { return "error"; }
			return get_host_name();
		} else if (tokens[0] == "get_api_version") {
			// BETA FUNCTION: semantics subject to change, use with caution
			// IN:  get_api_version
			// OUT: API version supported by this host
			if (!check_command_length(tokens, 0, 0)) { return "error"; }
			return get_api_version();
		} else if (tokens[0] == "get_api_support") {
			// BETA FUNCTION: semantics subject to change, use with caution
			// IN:  get_api_support
			// OUT: list of supported API features
			if (!check_command_length(tokens, 0, 0)) { return "error"; }
			return get_api_support();
		} else if (tokens[0] == "clock") {
			// BETA FUNCTION: semantics subject to change, use with caution
			// IN:  clock <num_cycles>
			// OUT: actual number of cycles stepped
			if (!check_command_length(tokens, 1, 1)) { return "error"; }
			int cycles = atoi(tokens[1].c_str());
		    module->propagate_changes();
		    for (int i=0; i<cycles; i++) {
		    	module->clock(dat_t<1>(0));
		    }
		    return itos(cycles);
		} else if (tokens[0] == "tick") {
			// BETA FUNCTION: semantics subject to change, use with caution
			// IN:  tick
			// OUT: ok (on success)
			// Update registers without propagation
			// updating registers.
		    module->clock_hi(dat_t<1>(0));
		    return "ok";
		} else if (tokens[0] == "propagate") {
			// BETA FUNCTION: semantics subject to change, use with caution
			// IN:  propagate
			// OUT: ok (on success)
			// This function propagates the combinational logic, without
			// updating registers.
			module->propagate_changes();
			return "ok";
		} else if (tokens[0] == "step") {
			// IN:  step <num_cycles>
			// OUT: actual number of cycles stepped
			if (!check_command_length(tokens, 1, 1)) { return "error"; }
			int n = atoi(tokens[1].c_str());
			module->propagate_changes();
			int ret = module->step(false, n);
			// Do we have print output to report?
			int nBytes = module->has_output();
			if (nBytes > 0) {
				cout << "PRINT" << " " << nBytes << " " << module->drain_output();
			}
			return itos(ret);
		} else if (tokens[0] == "set_clocks") {
			// BETA FUNCTION: semantics subject to change, use with caution
			// IN:  set_clocks
			// OUT: ???
			// I'm not really sure what this is supposed to do, but it was
			// in the old command API, so it's here now
		  std::vector< int > periods;
	        for (int i = 1; i < tokens.size(); i++) {
	          int period = atoi(tokens[i].c_str());
	          periods.push_back(period);
	        }
	        module->setClocks(periods);
	        return "ok";

		} else if (tokens[0] == "reset") {
			// IN:  reset <num_cycles>
			// OUT: actual number of cycles in reset
			if (!check_command_length(tokens, 0, 1)) { return "error"; }
			int cycles = 1;
			if (tokens.size() >= 2) {
				cycles = atoi(tokens[1].c_str());
			}
			for (int i=0; i<cycles; i++) {
			   	module->clock_lo(dat_t<1>(1));
			   	module->clock_hi(dat_t<1>(1));
		    }
		    module->clock_lo(dat_t<1>(0));
		    return itos(cycles);
		} else if (tokens[0] == "peek") {
			// LEGACY FUNCTION: do not use in new code
			// IN:  peek <node_name> | peek <mem_name> <mem_index>
			// OUT: value
			if (!check_command_length(tokens, 1, 2)) { return "error"; }
			cerr << "peek is deprecated, use wire_peek or mem_peek" << std::endl;
			module->propagate_changes();
			if (tokens.size() == 2) {
				return get_dat_by_name(tokens[1])->get_value();
			} else if (tokens.size() == 3) {
				return get_mem_by_name(tokens[1])->get_element(tokens[2]);
			}
		} else if (tokens[0] == "poke") {
			// LEGACY FUNCTION: do not use in new code
			// IN:  poke <node_name> <value> | poke <mem_name> <mem_index> <value>
			// OUT: true (on success), false (on failure)
			if (!check_command_length(tokens, 2, 3)) { return ""; }
			cerr << "poke is deprecated, use wire_poke or mem_poke" << std::endl;
			bool success;
			if (tokens.size() == 3) {
				success = get_dat_by_name(tokens[1])->set_value(tokens[2]);
			} else if (tokens.size() == 4) {
				success = get_mem_by_name(tokens[1])->set_element(tokens[2], tokens[3]);
			}
			std::string result;
			if (success) {
			  result = "true";
			  module->mark_stale();
			} else {
			  result = "false";
			}
			return result;
		} else if (tokens[0] == "wire_peek") {
			// IN:  wire_peek <node_name>
			// OUT: value
			if (!check_command_length(tokens, 1, 1)) { return "error"; }
			module->propagate_changes();
			return get_dat_by_name(tokens[1])->get_value();
		} else if (tokens[0] == "wire_poke") {
			// IN:  wire_poke <node_name> <value>
			// OUT: ok (on success)
			if (!check_command_length(tokens, 2, 2)) { return "error"; }
			bool success = get_dat_by_name(tokens[1])->set_value(tokens[2]);
			std::string result;
			if (success) {
			  result = "ok";
			  module->mark_stale();
			} else {
			  result = "error";
			}
			return result;
		} else if (tokens[0] == "mem_peek") {
			// IN:  mem_peek <mem_name> <mem_index>
			// OUT: value
			if (!check_command_length(tokens, 2, 2)) { return "error"; }
			module->propagate_changes();
			return get_mem_by_name(tokens[1])->get_element(tokens[2]);
		} else if (tokens[0] == "mem_poke") {
			// IN:  mem_poke <mem_name> <mem_index> <value>
			// OUT: ok (on success)
			if (!check_command_length(tokens, 3, 3)) { return "error"; }
			bool success = get_mem_by_name(tokens[1])->set_element(tokens[2], tokens[3]);
			std::string result;
			if (success) {
			  result = "ok";
			  module->mark_stale();
			} else {
			  result = "error";
			}
			return result;
			return success ? "ok" : "error";
		} else if (tokens[0] == "trace") {
			// IN:  trace n <node_name>+
			// OUT: values
                        // TODO: ADD MEM PEEK SUPPORT
                        stringstream ss;
			if (!check_command_length(tokens, 2)) { return "bad"; }
			int n = atoi(tokens[1].c_str());
                        for (int t = 0; t < n; t++) {
                          for (int i = 2; i < tokens.size(); i++) 
                            ss << " " << get_dat_by_name(tokens[i])->get_value();
                          int ret = module->step(false, 1);
                          // if (!ret)
                          //   return "error";
                        }
                        return ss.str();
		} else if (tokens[0] == "list_wires") {
			// IN:  list_wires
			// OUT: list of wires
			if (!check_command_length(tokens, 0, 0)) { return "error"; }
			std::string out = "";
			for (std::map<const char*, dat_api_base*>::iterator it = dat_table.begin(); it != dat_table.end(); it++) {
				out = out + it->second->get_pathname() + " ";
			}
			if (out.size() >= 1) {
				return out.substr(0, out.size() - 1);
			} else {
				return "";
			}
		} else if (tokens[0] == "list_mems") {
			// IN:  list_mems
			// OUT: list of memories
			if (!check_command_length(tokens, 0, 0)) { return "error"; }
			std::string out = "";
			for (std::map<const char*, mem_api_base*>::iterator it = mem_table.begin(); it != mem_table.end(); it++) {
				out = out + it->second->get_pathname() + " ";
			}
			if (out.size() >= 1) {
				return out.substr(0, out.size() - 1);
			} else {
				return "";
			}
		} else if (tokens[0] == "wire_width") {
			// IN:  wire_width <node>
			// OUT: bitwidth of wire
			if (!check_command_length(tokens, 1, 1)) { return "error"; }
			return get_dat_by_name(tokens[1])->get_width();
		} else if (tokens[0] == "mem_width") {
			// IN:  mem_width <node>
			// OUT: bitwidth of memory element
			if (!check_command_length(tokens, 1, 1)) { return "error"; }
			return get_mem_by_name(tokens[1])->get_width();
		} else if (tokens[0] == "mem_depth") {
			// IN:  mem_depth <node>
			// OUT: elements in memory
			if (!check_command_length(tokens, 1, 1)) { return "error"; }
			return get_mem_by_name(tokens[1])->get_depth();
		} else if (tokens[0] == "referenced_snapshot_save") {
			// BETA FUNCTION: semantics subject to change, use with caution
			// IN:  referenced_snapshot_save <name>
			// OUT: Reference name (an arbitrary string) for saved snapshot
			//      of current state, should be equivalent to the input.
			// Caution: the state may not be self-consistent (i.e. clk_lo
			// does not need to have been applied before this, and calls to
			// clk_lo immediately after restoring may change the state).
			if (!check_command_length(tokens, 1, 1)) { return "error"; }
			module->propagate_changes();
			mod_t *snapshot = module->clone();
			snapshot_table[tokens[1]] = snapshot;
			return tokens[1];
		} else if (tokens[0] == "referenced_snapshot_restore") {
			// BETA FUNCTION: semantics subject to change, use with caution
			// IN:  referenced_snapshot_restore <name>
			// OUT: ok (on success)
			if (!check_command_length(tokens, 1, 1)) { return "error"; }
			mod_t *snapshot = get_snapshot_by_reference(tokens[1]);
			if (snapshot == NULL) {	return "error";	}
			bool success = module->set_circuit_from(snapshot);
			std::string result;
			if (success) {
			  result = "ok";
			  module->mark_stale();
			} else {
			  result = "error";
			}
			return result;
		} else {
			std::cerr << "Unknown command: '" << tokens[0] << "'" << std::endl;
		}
		return "error";
	}

	void read_eval_print_loop() {
		while (true) {
		    std::string str_in;
		    do {
		    	std::getline(cin, str_in);
		    } while (cin.fail() && errno == EINTR);

		    if (!cin.good()) {
		    	break;
		    }

		    if (teefile != NULL) {
		        fprintf(teefile, "%s\n", str_in.c_str());
		        fflush(teefile);
		    }
		    if (str_in == "quit") {
		    	break;
		    } else {
		    	cout << eval_command(str_in) << std::endl;
		    }
		}
	}

protected:
	FILE* teefile;
	mod_t* module;

	// Mapping table functions
	virtual void init_mapping_table() = 0;

	dat_api_base* get_dat_by_name(std::string name) {
		if (dat_table.find(name.c_str()) != dat_table.end()) {
			return dat_table[name.c_str()];
		} else {
			std::cerr << "Unable to find dat '" << name << "'" << std::endl;
			return &this_dat_dummy;
		}
	}
	mem_api_base* get_mem_by_name(std::string name) {
		if (mem_table.find(name.c_str()) != mem_table.end()) {
			return mem_table[name.c_str()];
		} else {
			std::cerr << "Unable to find mem '" << name << "'" << std::endl;
			return &this_mem_dummy;
		}
	}

	mod_t* get_snapshot_by_reference(std::string name) {
		if (snapshot_table.find(name) != snapshot_table.end()) {
			return snapshot_table[name];
		} else {
			std::cerr << "Unable to find snapshot reference '" << name << "'" << std::endl;
			return NULL;
		}
	}

	class string_comparator {
	public:
		bool operator()(const char* x, const char* y) const {
			return strcmp(x, y) < 0;
		}
	};

	std::map<const char*, dat_api_base*, string_comparator> dat_table;
	std::map<const char*, mem_api_base*, string_comparator> mem_table;
	// TODO: replace the dummy with explicit NULL checks - this is simple
	// but a bit inelegant
	dat_dummy this_dat_dummy;
	mem_dummy this_mem_dummy;

	// Snapshot functions
	std::map<std::string, mod_t*> snapshot_table;
};

#pragma GCC diagnostic pop

#endif
