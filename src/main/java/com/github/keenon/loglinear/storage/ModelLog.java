package com.github.keenon.loglinear.storage;

import com.github.keenon.loglinear.model.GraphicalModel;

import java.io.*;
import java.util.*;

/**
 * Created by keenon on 1/12/16.
 *
 * The point of ModelLog is to open a streaming queue of data, backed by a store, which can be added to with minimal
 * IO. That optimizes for the common case, where we're recording a number of training examples to be used later
 * for training or analysis, but the examples are arriving in an online fashion.
 *
 * This is simpler than a ModelBatch, since it doesn't support deletion or editing post hoc. Files created by ModelLog
 * can be read by ModelBatch, and vice versa.
 */
public abstract class ModelLog extends ArrayList<GraphicalModel> {
    public boolean writeWithFactors = false;

    /**
     * This is the hook that children need to subclass out
     */
    public abstract void writeExample(GraphicalModel m) throws IOException;

    /**
     * Closes the backing store, which flushes any progress that's currently being cached.
     *
     * @throws IOException
     */
    public abstract void close() throws IOException;

    /**
     * Adds an element to the list, and if recordOnDisk is true, appends it to the backing store.
     *
     * @param m the model to add
     * @param recordOnDisk whether or not to add to the backing disk store
     * @return success
     */
    protected boolean add(GraphicalModel m, boolean recordOnDisk) {
        boolean success = super.add(m);
        if (!success) return false;
        if (recordOnDisk) {
            try {
                if (writeWithFactors) {
                    // Attempt to record this to the backing log file, with the factors
                    writeExample(m);
                } else {
                    // Attempt to record this to the backing log file, without the factors
                    Set<GraphicalModel.Factor> cachedFactors = m.factors;
                    m.factors = new HashSet<>();
                    writeExample(m);
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
