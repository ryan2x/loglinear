package com.github.keenon.loglinear.storage;

import com.github.keenon.loglinear.model.GraphicalModel;

import java.io.*;
import java.util.*;

/**
 * Created by keenon on 1/12/16.
 *
 * The point of ModelLog is to open a streaming queue of data, backed by a file, which can be added to with minimal
 * disk IO. That optimizes for the common case, where we're recording a number of training examples to be used later
 * for training or analysis, but the examples are arriving in an online fashion.
 *
 * This is simpler than a ModelBatch, since it doesn't support deletion or editing post hoc. Files created by ModelLog
 * can be read by ModelBatch, and vice versa.
 */
public class ModelLog extends ArrayList<GraphicalModel> {
    public boolean writeWithFactors = false;

    OutputStream os;

    /**
     * Creates a ModelLog that writes to 'filename'. It starts with any models already there.
     *
     * WARNING: behavior is undefined if multiple ModelLog's are backed by the same file. Do not do this.
     *
     * @param filename the file to back ModelLog, will be created if it doesn't exist yet
     * @throws IOException
     */
    public ModelLog(String filename) throws IOException {
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
    ModelLog(InputStream is, OutputStream os) throws IOException {
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

    /**
     * Adds an element to the list, and if recordOnDisk is true, appends it to the backing store.
     *
     * @param m the model to add
     * @param recordOnDisk whether or not to add to the backing disk store
     * @return success
     */
    private boolean add(GraphicalModel m, boolean recordOnDisk) {
        boolean success = super.add(m);
        if (!success) return false;
        if (recordOnDisk) {

            // Attempt to record this to the backing log file, without the factors

            try {
                if (writeWithFactors) {
                    m.writeToStream(os);
                } else {
                    Set<GraphicalModel.Factor> cachedFactors = m.factors;
                    m.factors = new HashSet<>();
                    m.writeToStream(os);
                    m.factors = cachedFactors;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return true;
    }

    @Override
    public boolean add(GraphicalModel m) {
        return add(m, true);
    }

    @Override
    public boolean remove(Object o) {
        throw new IllegalStateException("Operation unsupported by ModelLog. Try ModelBatch instead.");
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        throw new IllegalStateException("Operation unsupported by ModelLog. Try ModelBatch instead.");
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        throw new IllegalStateException("Operation unsupported by ModelLog. Try ModelBatch instead.");
    }

    @Override
    public void clear() {
        throw new IllegalStateException("Operation unsupported by ModelLog. Try ModelBatch instead.");
    }

    @Override
    public GraphicalModel set(int index, GraphicalModel element) {
        throw new IllegalStateException("Operation unsupported by ModelLog. Try ModelBatch instead.");
    }

    @Override
    public void add(int index, GraphicalModel element) {
        throw new IllegalStateException("Operation unsupported by ModelLog. Try ModelBatch instead.");
    }

    @Override
    public GraphicalModel remove(int index) {
        throw new IllegalStateException("Operation unsupported by ModelLog. Try ModelBatch instead.");
    }
}
