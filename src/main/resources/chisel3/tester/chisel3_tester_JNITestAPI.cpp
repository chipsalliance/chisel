#include "chisel3_tester_JNITestAPI.h"
#include "veri_api.h"
#include "sim_api.h"
#include <dlfcn.h>
#include <stdio.h>

struct SIMComponents {
    VCDInfo vcdInfo;
    SIM_API* api;
    char * sharedLibraryPath;
    void * sharedLibrary;
    typedef SIM_API * (*InitFunction)(VCDInfo& vcd);
    InitFunction ninit;
    typedef void (*FinishFunction)(SIM_API* apip, VCDInfo& vcd);
    FinishFunction nfinish;
    SIMComponents() : vcdInfo(), api(0), sharedLibraryPath(0), sharedLibrary(0), ninit(0), nfinish(0) {
//        fprintf(stderr, "SIMComponents:\n");
    }
};

//extern SIM_API * sim_jninit(VCDInfo& vcd);
//extern void sim_jnfinish(SIM_API* apip, VCDInfo& vcd);

//static SIMComponents* getSIMComponents(JNIEnv *env, jobject obj) {
//    jclass cls = env->GetObjectClass(obj);
//    jfieldID fid = env->GetFieldID(cls, "sim_api", "J");
//    SIMComponents* result = 0;
//    if (fid) {
//        result = (SIMComponents*) env->GetObjectField(obj, fid);
//    }
//    return result;
//}
//
//static void setSIMComponents(JNIEnv *env, jobject obj, SIMComponents* scp) {
//    jclass cls = env->GetObjectClass(obj);
//    jfieldID fid = env->GetFieldID(cls, "sim_api", "J");
//    if (fid) {
//        env->SetLongField(obj, fid, (jlong) scp);
//    }
//}

static void unload(SIMComponents* scp) {
    if (scp->sharedLibrary) {
        int unload = dlclose(scp->sharedLibrary);
        if (unload == -1) {
            fprintf(stderr, "JNITestAPI.unload: couldn't unload %s: %s\n", scp->sharedLibraryPath, dlerror());
        }
    }
    free((void *)scp->sharedLibraryPath);
    scp->sharedLibraryPath = 0;
    free((void *)scp->vcdInfo.fileName);
    scp->vcdInfo.fileName = 0;
    delete scp;
}

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    ninit
 * Signature: (Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_chisel3_tester_JNITestAPI_ninit
  (JNIEnv *env, jobject obj, jstring jsharedLibraryPath, jstring jvcdFileName) {
    SIMComponents* scp = new SIMComponents;
//    fprintf(stderr, "JNITestAPI.ninit\n");
    const char * vcdFileName = env->GetStringUTFChars(jvcdFileName, NULL);
    const char * sharedLibraryPath = env->GetStringUTFChars(jsharedLibraryPath, NULL);
//    fprintf(stderr, "JNITestAPI: args %s, %s\n", sharedLibraryPath, vcdFileName);
    scp->vcdInfo.fileName = strdup(vcdFileName);
    scp->sharedLibraryPath = strdup(sharedLibraryPath);
//    fprintf(stderr, "JNITestAPI: loading %s, %s\n", scp->sharedLibraryPath, scp->vcdInfo.fileName);
    scp->sharedLibrary = 0;
    scp->sharedLibrary = dlopen(sharedLibraryPath, RTLD_NOW);
    bool ok = true;
    if (scp->sharedLibrary == 0) {
        fprintf(stderr, "JNITestAPI: couldn't load %s: %s\n", scp->sharedLibraryPath, dlerror());
        ok = false;
    } else {
        struct {
            const char * name;
            void * value;
        } symbols[] = {{"_Z10sim_jninitR7VCDInfo", 0}, {"_Z12sim_jnfinishP7SIM_APIR7VCDInfo", 0}};
        for (int i = 0; i < sizeof(symbols)/sizeof(symbols[0]); i += 1) {
            symbols[i].value = dlsym(scp->sharedLibrary, symbols[i].name);
            if (symbols[i].value == 0) {
                fprintf(stderr, "JNITestAPI: couldn't find \"%s\": %s\n", symbols[i].name, dlerror());
                ok = false;
            }
        }
        if (ok) {
            scp->ninit = (SIMComponents::InitFunction) symbols[0].value;
            scp->nfinish = (SIMComponents::FinishFunction) symbols[1].value;
            scp->api = scp->ninit(scp->vcdInfo);
        }
    }
    env->ReleaseStringUTFChars(jsharedLibraryPath, sharedLibraryPath);
    env->ReleaseStringUTFChars(jvcdFileName, vcdFileName);
    if (!ok) {
        unload(scp);
        scp = 0;
    }
    return (jlong)scp;
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    nreset
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_nreset
  (JNIEnv *env, jobject obj, jlong jscp) {
    SIMComponents* scp = (SIMComponents*) jscp;
    scp->api->reset();
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    nstart
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_nstart
  (JNIEnv *env, jobject obj, jlong jscp) {
    SIMComponents* scp = (SIMComponents*) jscp;
    scp->api->start();
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    nstep
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_nstep
  (JNIEnv *env, jobject obj, jlong jscp, jint ncycles) {
    SIMComponents* scp = (SIMComponents*) jscp;
    scp->api->step();
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    nupdate
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_nupdate
  (JNIEnv *env, jobject obj, jlong jscp) {
    SIMComponents* scp = (SIMComponents*) jscp;
    scp->api->update();
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    ngetID
 * Signature: (JLjava/lang/String;)J
 */
JNIEXPORT jint JNICALL Java_chisel3_tester_JNITestAPI_ngetID
  (JNIEnv *env, jobject obj, jlong jscp, jstring jsignalPath) {
    SIMComponents* scp = (SIMComponents*) jscp;
    const char *signalPath = env->GetStringUTFChars(jsignalPath, NULL);
    jint result = scp->api->search(signalPath);
    env->ReleaseStringUTFChars(jsignalPath, signalPath);
    return result;
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    ngetChunk
 * Signature: (JJ)J
 */
JNIEXPORT jint JNICALL Java_chisel3_tester_JNITestAPI_ngetSignalWordSize
  (JNIEnv *env, jobject obj, jlong jscp, jint id) {
    SIMComponents* scp = (SIMComponents*) jscp;
    return scp->api->getSignalWordSize(id);
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    npeekn
 * Signature: (JII[J)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_npeekn
  (JNIEnv *env, jobject obj, jlong jscp, jint id, jint n, jlongArray jdata) {
    SIMComponents* scp = (SIMComponents*) jscp;
    uint64_t * datau64 = (uint64_t*) env->GetLongArrayElements(jdata, NULL);
    scp->api->peekn(id, n, datau64);
    env->ReleaseLongArrayElements(jdata, (jlong *)datau64, 0);
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    npoken
 * Signature: (JII[J)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_npoken
  (JNIEnv *env, jobject obj, jlong jscp, jint id, jint n, jlongArray jdata) {
    SIMComponents* scp = (SIMComponents*) jscp;
    uint64_t * datau64 = (uint64_t*) env->GetLongArrayElements(jdata, NULL);
    scp->api->poken(id, n, datau64);
    env->ReleaseLongArrayElements(jdata, (jlong *)datau64, JNI_ABORT);
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    nsetInputSignals
 * Signature: (JLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_nsetInputSignals
  (JNIEnv *env, jobject obj, jlong jscp, jobject jByteBuffer) {
    SIMComponents* scp = (SIMComponents*) jscp;
//    uint64_t * datau64 = (uint64_t *)env->GetDirectBufferAddress(jByteBuffer);
//    scp->api->setInputSignals(datau64);
      scp->api->recv_tokens();
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    ngetOutputSignals
 * Signature: (JLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_ngetOutputSignals
  (JNIEnv *env, jobject obj, jlong jscp, jobject jByteBuffer) {
    SIMComponents* scp = (SIMComponents*) jscp;
//    uint64_t * datau64 = (uint64_t *)env->GetDirectBufferAddress(jByteBuffer);
//    scp->api->getOutputSignals(datau64);
      scp->api->send_tokens();
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    ninitChannels
 * Signature: (JLjava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_ninitChannels
  (JNIEnv *env, jobject obj, jlong jscp, jobject jinputByteBuffer, jobject joutputByteBuffer) {
    SIMComponents* scp = (SIMComponents*) jscp;
//    std::cerr << std::hex << "Java_chisel3_tester_JNITestAPI_ninitChannels: in 0x" << jinputByteBuffer << ", out 0x" << joutputByteBuffer << std::endl;
    volatile char * indatau64 = (volatile char *)env->GetDirectBufferAddress(jinputByteBuffer);
    volatile char * outdatau64 = (volatile char *)env->GetDirectBufferAddress(joutputByteBuffer);
//    std::cerr << std::hex << "Java_chisel3_tester_JNITestAPI_ninitChannels: in 0x" << indatau64 << ", out 0x" << outdatau64 << std::endl;
    scp->api->init_channels(indatau64, outdatau64, (volatile char *)0);
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    ngetInputBuffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_chisel3_tester_JNITestAPI_ngetInputBuffer
  (JNIEnv *env, jobject obj, jlong jscp) {
    SIMComponents* scp = (SIMComponents*) jscp;
    void * address = 0;
    long size = 0;
//    fprintf(stderr, "Java_chisel3_tester_JNITestAPI_ngetInputBuffer: scp 0x%lx, api 0x%lx\n", scp, scp->api);
    scp->api->getInputChannelBuffer(address, size);
//    std::cerr << std::hex << "Java_chisel3_tester_JNITestAPI_ngetInputBuffer: address 0x" << address << ", size 0x" << size << std::endl;
    jobject byteBuffer = env->NewDirectByteBuffer(address, size);
    return byteBuffer;
  }


/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    ngetOutputBuffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_chisel3_tester_JNITestAPI_ngetOutputBuffer
  (JNIEnv *env, jobject obj, jlong jscp) {
    SIMComponents* scp = (SIMComponents*) jscp;
    void * address = 0;
    long size = 0;
    scp->api->getOutputChannelBuffer(address, size);
//    std::cerr << std::hex << "Java_chisel3_tester_JNITestAPI_ngetOutputBuffer: address 0x" << address << ", size 0x" << size << std::endl;
    jobject byteBuffer = env->NewDirectByteBuffer(address, size);
    return byteBuffer;
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    nfinish
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_nfinish
  (JNIEnv *env, jobject obj, jlong jscp) {
    SIMComponents* scp = (SIMComponents*) jscp;
    scp->nfinish(scp->api, scp->vcdInfo);
    unload(scp);
  }

/*
 * Class:     chisel3_tester_JNITestAPI
 * Method:    nabort
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_chisel3_tester_JNITestAPI_nabort
  (JNIEnv *env, jobject obj, jlong jscp) {
    SIMComponents* scp = (SIMComponents*) jscp;
    unload(scp);
  }
