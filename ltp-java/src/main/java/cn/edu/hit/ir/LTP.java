package cn.edu.hit.ir;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URL;
import java.util.List;
import java.util.ArrayList;
import java.io.File;
import java.nio.file.Files;

import static java.lang.System.mapLibraryName;


public class LTP {
    static {
        String libName = "ltp4j";
        try {
            System.loadLibrary(libName);
        } catch (UnsatisfiedLinkError ex) {
            URL url = LTP.class.getClassLoader().getResource(mapLibraryName(libName));
            try {
                File file = Files.createTempFile("jni", libName).toFile();
                file.deleteOnExit();
                file.delete();
                try (InputStream in = url.openStream()) {
                    Files.copy(in, file.toPath());
                }
                System.load(file.getCanonicalPath());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

    }

    // 指向底层对象的指针，伪装成Java的long
    private final long ptr;

    public LTP(String path) throws Exception {
        ptr = rust_init(path);
    }

    // 在finalize里释放
    public void finalize() {
        rust_release(ptr);
    }

    public List pipeline(List sentences) throws Exception {
        return rust_pipeline(ptr, sentences);
    }


    public static void main(String[] args) {
        LTP ltp = null;
        try {
            ltp = new LTP("onnx-small");
            ArrayList<String> inputs = new ArrayList<>();
            inputs.add("他叫汤姆去拿外衣！");
            List output = ltp.pipeline(inputs);
            System.out.println(output);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static native long rust_init(String path) throws Exception;

    private static native void rust_release(long ptr);

    private native List rust_pipeline(long ptr, List sentences) throws Exception;
}