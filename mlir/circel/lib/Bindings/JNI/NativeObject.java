package CircelJNI;

import java.lang.ref.Cleaner;

/**
 * A Java Object backed by a C++ NativeReference.
 */
abstract class NativeObject implements AutoCloseable {
	private long nativeReference;
	private Cleaner.Cleanable cleanable;
	private static final Cleaner sharedCleaner = Cleaner.create();

	public void close() {
		cleanable.clean();
	}

	/**
	 * Releases a NativeReference.
	 */
	private static class ReleaseNativeReference implements Runnable {
		private long reference;

		public ReleaseNativeReference(long reference) {
			this.reference = reference;
		}

		public void run() {
			NativeObject.releaseNativeReference(reference);
		}
	}

	private static native void releaseNativeReference(long nativeReference);

	protected NativeObject(long nativeReference) {
		this.nativeReference = nativeReference;
		this.cleanable = sharedCleaner.register(this, new ReleaseNativeReference(nativeReference));
	}
}
