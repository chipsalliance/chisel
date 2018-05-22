package chisel3.tester;

import java.lang.reflect.*;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;


/**
 * An invocation handler that passes on any calls made to it directly to its delegate.
 * This is useful to handle identical classes loaded in different classloaders - the
 * VM treats them as different classes, but they have identical signatures.
 *
 * Note this is using class.getMethod, which will only work on public methods.
 */
class PassThroughProxyHandler implements InvocationHandler {
    private final Object delegate;
    public PassThroughProxyHandler(Object delegate) {
        this.delegate = delegate;
    }
    public Object invoke(Object proxy, Method method, Object[] args)
            throws Throwable {
        Method delegateMethod = delegate.getClass().getMethod(method.getName(), method.getParameterTypes());
        return delegateMethod.invoke(delegate, args);
    }
}

public final class SingletonLoaderShim implements SingletonLoaderShimInterface {
    private static SingletonLoaderShimInterface instance = null;
    private static Boolean loading = true;
    public synchronized static SingletonLoaderShimInterface getInstance() {
        ClassLoader myClassLoader = SingletonLoaderShim.class.getClassLoader();
        ClassLoader tryClassLoader = myClassLoader;
        if (instance==null) {
            // The root classloader is sun.misc.Launcher package. If we are not in a sun package,
            // we need to get hold of the instance of ourself from the class in the root classloader.
            while (!tryClassLoader.toString().startsWith("sun.")) {
                ClassLoader parentClassLoader = tryClassLoader.getParent();
                System.out.println("SingletonLoaderShim: tryClassLoader " + tryClassLoader.toString() + ", parent " + parentClassLoader.toString());
                if (parentClassLoader == null)
                    break;
                tryClassLoader = parentClassLoader;
            }
            System.out.println("SingletonLoaderShim: tryClassLoader " + tryClassLoader.toString());
            if (!(loading /*|| tryClassLoader.toString().startsWith("sun.")*/)) {
                try {
                    loading = true;
                    // So we find our parent classloader
                    // And get the other version of our current class
                    //                    Class otherClassInstance = Class.forName(SingletonLoaderShim.class.getName(), true, parentClassLoader);
                    //                    // And call its getInstance method - this gives the correct instance of ourself
                    //                    Method getInstanceMethod = otherClassInstance.getDeclaredMethod("getInstance");
                    Class otherClassInstance = tryClassLoader.loadClass(SingletonLoaderShim.class.getName());
                    // And call its getInstance method - this gives the correct instance of ourself
                    Method getInstanceMethod = otherClassInstance.getDeclaredMethod("getInstance", new Class[]{});
                    Object otherSingletonLoaderShim = getInstanceMethod.invoke(null, new Object[]{});
                    // But, we can't cast it to our own interface directly because classes loaded from
                    // different classloaders implement different versions of an interface.
                    // So instead, we use java.lang.reflect.Proxy to wrap it in an object that *does*
                    // support our interface, and the proxy will use reflection to pass through all calls
                    // to the object.
                    instance = (SingletonLoaderShimInterface) Proxy.newProxyInstance(tryClassLoader,
                            new Class[]{SingletonLoaderShimInterface.class},
                            new PassThroughProxyHandler(otherSingletonLoaderShim));
                    // And catch the usual tedious set of reflection exceptions
                    // We're cheating here and just catching everything - don't do this in real code
                    System.out.println("SingletonLoaderShim: old instance, class loader: " + tryClassLoader.toString());
                } catch (Exception e) {
                    e.printStackTrace();
                    // Give up and use this instance
                    System.out.println("SingletonLoaderShim: (try) new instance, class loader: " + tryClassLoader.toString());
                    instance = new SingletonLoaderShim();
                }
            } else {
                // We're in the root classloader, so the instance we have here is the correct one
                System.out.println("SingletonLoaderShim: (top) new instance, class loader: " + myClassLoader.toString());
                instance = new SingletonLoaderShim();
            }
        } else {
            System.out.println("SingletonLoaderShim: (old) new instance, class loader: " + myClassLoader.toString());
        }

        return instance;
    }
    private static Boolean shimLoaded;
    // Defeat instantiation by anything outside of this class.
    private SingletonLoaderShim() {
        shimLoaded = false;
    }
    public synchronized final void loadJNITestShim(String fullPath) throws Exception {
        if (true) {
            java.lang.reflect.Method m = ClassLoader.class.getDeclaredMethod("findLoadedClass", new Class[] { String.class });
            m.setAccessible(true);
            ClassLoader cl = ClassLoader.getSystemClassLoader();
            Object test1 = m.invoke(cl, "chisel3.tester$JNITestAPI");
            if (!shimLoaded) {
                System.out.println("loading shim " + fullPath + ", found " + test1 != null);
                shimLoaded = true;
                System.load(fullPath);
            } else {
                System.out.println("not loading shim " + fullPath + ", found " + test1 != null);
            }
        } else {

        }
    }
}
