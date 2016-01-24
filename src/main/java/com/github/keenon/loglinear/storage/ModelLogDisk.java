package com.github.keenon.loglinear.storage;

import com.github.keenon.loglinear.model.GraphicalModel;

import java.io.*;
import java.util.HashSet;
import java.util.Set;

/**
 * Created by keenon on 1/23/16.
 *
 * Implements a ModelLog that lives on disk
 */
public class ModelLogDisk extends ModelLog {
    OutputStream os;

    /**
     * Creates a ModelLog that writes to 'filename'. It starts with any models already there.
     *
     * WARNING: behavior is undefined if multiple ModelLog's are backed by the same file. Do not do this.
     *
     * @param filename the file to back ModelLog, will be created if it doesn't exist yet
     * @throws IOException
     */
    public ModelLogDisk(String filename) throws IOException {
        if (filename.contains("/")) {
            String path = filename.substring(0, filename.lastIndexOf("/"));
            File dir = new File(path);
            if (!dir.exists()) dir.mkdirs();
        }
        File f = new File(filename);
        System.err.println(filename);
        if (!f.exists()) {
            f.createNewFile();
        }

        InputStream is = new FileInputStream(filename);
        GraphicalModel read;
        while ((read = GraphicalModel.readFromStream(is)) != null) {
            add(read, false);
        }
        is.close();
        os = new FileOutputStream(filename, true); // second arg is "append", prevents restarting file
    }

    /**
     * This is a constructor that's basically only for internal use (testing).
     * @param is the input stream to read the existing models from
     * @param os the output stream to write the models to
     */
    ModelLogDisk(InputStream is, OutputStream os) throws IOException {
        GraphicalModel read;
        while ((read = GraphicalModel.readFromStream(is)) != null) {
            add(read, false);
        }
        this.os = os;
    }

    /**
     * Closes the file OutputStream to the backing store, which flushes any progress that's currently being cached.
     *
     * @throws IOException
     */
    public void close() throws IOException {
        os.close();
    }

    @Override
    public void writeExample(GraphicalModel m) throws IOException {
        m.writeToStream(os);
        os.flush();
    }
}
