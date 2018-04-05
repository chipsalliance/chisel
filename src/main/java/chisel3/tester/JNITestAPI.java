package chisel3.tester;
import java.nio.ByteBuffer;

public final class JNITestAPI {
    // The underlying native API.
    private long sim_api;
    // Buffers allocated in C++ code.
    public ByteBuffer inputBuffer;
    public ByteBuffer outputBuffer;
    private native long ninit(String sharedLibraryPath, String vcdFileName);
    private native void nreset(long sim);
    private native void nstart(long sim);
    private native void nstep(long sim, int cycles);
    private native void nupdate(long sim);
    private native int ngetID(long sim, String signalPath);
    private native int ngetSignalWordSize(long sim, int signalID);
    private native void npeekn(long sim, int signalID, int n, long data[]);
    private native void npoken(long sim, int signalID, int n, long data[]);
    private native void nsetInputSignals(long sim, java.nio.ByteBuffer bb);
    private native void ngetOutputSignals(long sim, java.nio.ByteBuffer bb);
    private native void ninitChannels(long sim, java.nio.ByteBuffer input, java.nio.ByteBuffer output);
    private native ByteBuffer ngetInputBuffer(long sim);
    private native ByteBuffer ngetOutputBuffer(long sim);
    private native void nfinish(long sim);
    private native void nabort(long sim);

    public JNITestAPI() {
        sim_api = 0;
        inputBuffer = null;
        outputBuffer = null;
    }
    public void init(String sharedLibraryPath, String vcdFileName) {
        // Load the shared library implementation.
        //        val currentDir = System.getProperty("user.dir")
        //        System.load(s"${currentDir}/src/test/c/JNITest.dylib")
        //        val dependentLibrarys = Seq("/usr/lib/libSystem.B.dylib", "/usr/lib/libc++.1.dylib")
        //        val javaLibraryPath = System.getProperty("java.library.path")
        //        System.setProperty("java.library.path", javaLibraryPath + ":/usr/lib")
        //        for (lib <- dependentLibrarys) {
        //          System.load(lib)
        //        }
        //System.load(sharedLibraryPath);
        sim_api = ninit(sharedLibraryPath, vcdFileName);
        if (sim_api == 0)
            throw new RuntimeException("JNITestAPI: load failed.");
        inputBuffer = ngetInputBuffer(sim_api);
        outputBuffer = ngetOutputBuffer(sim_api);
    }
    public void initChannels(java.nio.ByteBuffer input, java.nio.ByteBuffer output) {
        ninitChannels(sim_api, input, output);
    }
    public void reset(int cycles) {
        for (int i = 0; i < cycles; i += 1) {
            nreset(sim_api);
            nstart(sim_api);
        }
    }
    public void start() {
        nstart(sim_api);
    }
    public void step(int cycles) {
        nstep(sim_api, cycles);
    }
    public void update() {
        nupdate(sim_api);
    }
    public int getID(String signalPath) {
        return ngetID(sim_api, signalPath);
    }
    public int getSignalWordSize(int signalID) {
        return ngetSignalWordSize(sim_api, signalID);
    }
    public void peekn(int id, int n, long data[]) {
        npeekn(sim_api, id, n, data);
    }
    public void poken(int id, int n, long data[]) {
        npoken(sim_api, id, n, data);
    }
    public void setInputSignals(java.nio.ByteBuffer bb) {
        byte[] bytes = new byte[8];
        for (int i = 0; i < 4; i += 1) {
            bytes[i] = bb.get(i);
        }
//        System.err.printf("setInputSignals: %d, %d, %d, %d\n", bytes[0], bytes[1], bytes[2], bytes[3]);
        nsetInputSignals(sim_api, bb);
    }
    public void getOutputSignals(java.nio.ByteBuffer bb) {
        ngetOutputSignals(sim_api, bb);
        byte[] bytes = new byte[8];
        for (int i = 0; i < 5; i += 1) {
            bytes[i] = bb.get(i);
        }
//        System.err.printf("getOutputSignals: %d, %d, %d, %d, %d\n", bytes[0], bytes[1], bytes[2], bytes[3], bytes[4]);
    }
    public void finish() {
        long sa = sim_api;
        sim_api = 0;
        if (sa != 0)
            nfinish(sa);
    }
    public void abort() {
        long sa = sim_api;
        sim_api = 0;
        if (sa != 0)
            nabort(sa);
    }
}
