package com.github.keenon.loglinear.model;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by keenon on 10/20/15.
 *
 * This is a wrapper function to keep a namespace of namespace of recognized features, so that building a set of
 * ConcatVectors for featurizing a model is easier and more intuitive. It's actually quite simple, and threadsafe.
 */
public class ConcatVectorNamespace {
    final Map<String,Integer> featureToIndex = new HashMap<>();

    /**
     * Creates a new vector that is appropriately sized to accommodate all the features that have been named so far.
     * @return a new, empty ConcatVector
     */
    public ConcatVector newVector() {
        return new ConcatVector(featureToIndex.size());
    }

    /**
     * An optimization, this lets clients inform the ConcatVectorNamespace of how many features to expect, so
     * that we can avoid resizing ConcatVectors.
     * @param featureName the feature to add to our index
     */
    public void ensureFeature(String featureName) {
        synchronized (featureToIndex) {
            if (!featureToIndex.containsKey(featureName)) {
                featureToIndex.put(featureName, featureToIndex.size());
            }
        }
    }

    /**
     * This adds a dense feature to a vector, setting the appropriate component of the given vector to the passed in
     * value.
     * @param vector the vector
     * @param featureName the feature whose value to set
     * @param value the value we want to set this vector to
     */
    public void setDenseFeature(ConcatVector vector, String featureName, double[] value) {
        ensureFeature(featureName);
        vector.setDenseComponent(featureToIndex.get(featureName), value);
    }

    /**
     * This adds a sparse feature to a vector, setting the appropriate component of the given vector to the passed in
     * value.
     * @param vector the vector
     * @param featureName the feature whose value to set
     * @param index the index of the one-hot vector to set
     * @param value the value we want to set this one-hot index to
     */
    public void setSparseFeature(ConcatVector vector, String featureName, int index, double value) {
        ensureFeature(featureName);
        vector.setSparseComponent(featureToIndex.get(featureName), index, value);
    }
}
