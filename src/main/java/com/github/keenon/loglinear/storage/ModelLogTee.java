package com.github.keenon.loglinear.storage;

import com.github.keenon.loglinear.model.GraphicalModel;

import java.io.IOException;

/**
 * Created by keenon on 1/23/16.
 *
 * This logs to two different ModelLogs, though it only reads from the first. Otherwise this functions just like a regular
 * ModelLog. Think of the "tee" command in Unix.
 */
public class ModelLogTee extends ModelLog {
    ModelLog main;
    ModelLog shadow;

    /**
     * This creates a ModelLog by reading the contents of the first ModelLog passed in. All writes get copied to both
     * ModelLog's though.
     * @param main the ModelLog to read/write from
     * @param shadow the ModelLog to just write to
     */
    public ModelLogTee(ModelLog main, ModelLog shadow) {
        this.main = main;
        this.shadow = shadow;
        for (GraphicalModel m : main) add(m, false);
    }

    @Override
    public void writeExample(GraphicalModel m) throws IOException {
        main.add(m);
        shadow.add(m);
    }

    @Override
    public void close() throws IOException {
        main.close();
        shadow.close();
    }
}
