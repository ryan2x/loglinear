package com.github.keenon.loglinear.fastmath;

/**
 * Created by keenon on 9/20/15.
 *
 * Provides optional native acceleration for math operations, if the native library is on the DLL
 */
public class FastMath {
    private static boolean loadedNative = false;

    public static double log(double d) {
        if (loadedNative) return native_log(d);
        return Math.log(d);
    }

    public static double exp(double d) {
        if (loadedNative) return native_exp(d);
        return Math.exp(d);
    }

    public static native double native_log(double d);
    public static native double native_exp(double d);

    static {
        try {
            System.load("libmath.so");
            loadedNative = true;
        }
        catch (UnsatisfiedLinkError e) {
            System.err.println("WARN: Could not load native math acceleration.");
        }
    }
}
