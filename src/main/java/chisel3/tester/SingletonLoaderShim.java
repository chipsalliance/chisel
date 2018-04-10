package chisel3.tester;

public final class SingletonLoaderShim {
    public final static SingletonLoaderShim INSTANCE = new SingletonLoaderShim();
    private static Boolean shimLoaded;
    // Defeat instantiation by anything outside of this class.
    private SingletonLoaderShim() {
        shimLoaded = false;
    }
    public synchronized final void loadJNITestShim(String fullPath) {
        if (!shimLoaded) {
            shimLoaded = true;
            System.load(fullPath);
        }
    }
}
