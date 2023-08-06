#include "CircelJNI_NativeObject.h"

#include "Support.h"
#include "circel/Bindings/Support/NativeReference.h"
#include <mlir/IR/BuiltinAttributes.h>

JNIEXPORT void JNICALL
Java_CircelJNI_NativeObject_releaseNativeReference(JNIEnv *, jclass, jlong) {}
