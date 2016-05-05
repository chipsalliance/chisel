#ifndef __VERILATOR_API__
#define __VERILATOR_API__

#include "verilated.h"
#include "sim_api.h"
#include <vector>

class VerilatorDataWrapper {
public:
    virtual size_t get_value(uint64_t* values) = 0;
    virtual size_t put_value(uint64_t* values) = 0;
    virtual size_t get_num_words() = 0;
};

class VerilatorCData : public VerilatorDataWrapper {
public:
    VerilatorCData(CData* _signal) {
        signal = _signal;
    }
    virtual size_t get_value(uint64_t* values) {
        values[0] = (uint64_t) (*signal);
        return 1;
    }
    virtual size_t put_value(uint64_t* values) {
        uint64_t mask = 0xff;
        *signal = (CData) (mask & values[0]);
        return 1;
    }
    virtual size_t get_num_words() {
        return 1;
    }
private:
    CData* signal;
};

class VerilatorSData : public VerilatorDataWrapper {
public:
    VerilatorSData(SData* _signal) {
        signal = _signal;
    }
    virtual size_t get_value(uint64_t* values) {
        values[0] = (uint64_t) (*signal);
        return 1;
    }
    virtual size_t put_value(uint64_t* values) {
        uint64_t mask = 0xffff;
        *signal = (SData) (mask & values[0]);
        return 1;
    }
    virtual size_t get_num_words() {
        return 1;
    }
private:
    SData* signal;
};

class VerilatorIData : public VerilatorDataWrapper {
public:
    VerilatorIData(IData* _signal) {
        signal = _signal;
    }
    virtual size_t get_value(uint64_t* values) {
        values[0] = (uint64_t) (*signal);
        return 1;
    }
    virtual size_t put_value(uint64_t* values) {
        uint64_t mask = 0xffffffff;
        *signal = (IData) (mask & values[0]);
        return 1;
    }
    virtual size_t get_num_words() {
        return 1;
    }
private:
    IData* signal;
};

class VerilatorQData : public VerilatorDataWrapper {
public:
    VerilatorQData(QData* _signal) {
        signal = _signal;
    }
    virtual size_t get_value(uint64_t* values) {
        values[0] = (uint64_t) (*signal);
        return 1;
    }
    virtual size_t put_value(uint64_t* values) {
        *signal = (QData) values[0];
        return 1;
    }
    virtual size_t get_num_words() {
        return 1;
    }
private:
    QData* signal;
};

class VerilatorWData : public VerilatorDataWrapper {
public:
    VerilatorWData(WData* _wdatas, size_t _numWdatas) {
        wdatas = _wdatas;
        numWdatas = _numWdatas;
    }
    virtual size_t get_value(uint64_t* values) {
        bool numWdatasEven = (numWdatas % 2) == 0;
        for(int i = 0; i < numWdatas/2; i++) {
            uint64_t value = ((uint64_t) wdatas[i*2 + 1]) << 32 | wdatas[i*2];
            values[i] = value;
        }
        if (!numWdatasEven) {
            values[numWdatas/2] = wdatas[numWdatas - 1];
        }
        return get_num_words();
    }
    virtual size_t put_value(uint64_t* values) {
        bool numWdatasEven = (numWdatas % 2) == 0;
        for(int i = 0; i < numWdatas/2; i++) {
            wdatas[i*2] = values[i];
            wdatas[i*2 + 1] = values[i] >> 32;
        }
        if (!numWdatasEven) {
            wdatas[numWdatas - 1] = values[numWdatas/2];
        }
        return get_num_words();
    }
    virtual size_t get_num_words() {
        bool numWdatasEven = numWdatas % 2 == 0;
        if (numWdatasEven){
            return numWdatas/2;
        } else {
            return numWdatas/2 + 1;
        }
    }
private:
    WData* wdatas;
    size_t numWdatas;
};

#endif
