package com.github.keenon.loglinear.simple;

import com.github.keenon.loglinear.model.GraphicalModel;

/**
 * Created by keenon on 1/13/16.
 *
 * We have this interface so that users of these SimpleModel's (like LENSE) can have a unified interface to talk to
 * several different kinds of SimpleModel.
 */
public abstract class SimpleModel<T> {
    public abstract GraphicalModel getFeaturizedModel(T t);
    public abstract void addTrainingExample(GraphicalModel model);
}
