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
    final Map<String, Map<String,Integer>> sparseFeatureIndex = new HashMap<>();

    /**
     * Creates a new vector that is appropriately sized to accommodate all the features that have been named so far.
     * @return a new, empty ConcatVector
     */
    public ConcatVector newVector() {
        return new ConcatVector(featureToIndex.size());
    }

    /**
     * This constructs a fresh vector that is sized correctly to accommodate all the known sparse values for vectors
     * that are possibly sparse.
     *
     * @return a new, internally correctly sized ConcatVector that will work correctly as weights for features from
     *         this namespace;
     */
    public ConcatVector newWeightsVector() {
        ConcatVector vector = new ConcatVector(featureToIndex.size());
        for (String s : sparseFeatureIndex.keySet()) {
            int size = sparseFeatureIndex.get(s).size();
            vector.setDenseComponent(ensureFeature(s), new double[size]);
        }
        return vector;
    }

    /**
     * An optimization, this lets clients inform the ConcatVectorNamespace of how many features to expect, so
     * that we can avoid resizing ConcatVectors.
     * @param featureName the feature to add to our index
     */
    public int ensureFeature(String featureName) {
        synchronized (featureToIndex) {
            if (!featureToIndex.containsKey(featureName)) {
                featureToIndex.put(featureName, featureToIndex.size());
            }
            return featureToIndex.get(featureName);
        }
    }

    /**
     * An optimization, this lets clients inform the ConcatVectorNamespace of how many sparse feature components to
     * expect, again so that we can avoid resizing ConcatVectors.
     * @param featureName the feature to use in our index
     * @param index the sparse value to ensure is available
     */
    public int ensureSparseFeature(String featureName, String index) {
        ensureFeature(featureName);
        synchronized (sparseFeatureIndex) {
            if (!sparseFeatureIndex.containsKey(featureName)) {
                sparseFeatureIndex.put(featureName, new HashMap<>());
            }
        }
        final Map<String,Integer> sparseIndex = sparseFeatureIndex.get(featureName);
        synchronized (sparseIndex) {
            if (!sparseIndex.containsKey(index)) {
                sparseIndex.put(index, sparseIndex.size());
            }
            return sparseIndex.get(index);
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
        vector.setDenseComponent(ensureFeature(featureName), value);
    }

    /**
     * This adds a sparse feature to a vector, setting the appropriate component of the given vector to the passed in
     * value.
     * @param vector the vector
     * @param featureName the feature whose value to set
     * @param index the index of the one-hot vector to set, as a string, which we will translate into a mapping
     * @param value the value we want to set this one-hot index to
     */
    public void setSparseFeature(ConcatVector vector, String featureName, String index, double value) {
        vector.setSparseComponent(ensureFeature(featureName), ensureSparseFeature(featureName, index), value);
    }

    /**
     * This prints out a ConcatVector by mapping to the namespace, to make debugging learning algorithms easier.
     *
     * @param vector the vector to print
     * @return a flat string that can be printed to the console or stored in a log
     */
    public String debugVector(ConcatVector vector) {
        StringBuilder sb = new StringBuilder();

        for (String key : featureToIndex.keySet()) {
            sb.append(key).append(":\n");
            int i = featureToIndex.get(key);
            if (vector.isComponentSparse(i)) {
                debugFeatureValue(key, vector.getSparseIndex(i), vector, sb);
            }
            else {
                double[] arr = vector.getDenseComponent(i);
                for (int j = 0; j < arr.length; j++) {
                    debugFeatureValue(key, j, vector, sb);
                }
            }
        }
        return sb.toString();
    }

    /**
     * This writes a feature's individual value, using the human readable name if possible, to a StringBuilder
     */
    private void debugFeatureValue(String feature, int index, ConcatVector vector, StringBuilder sb) {
        sb.append("\t");
        if (sparseFeatureIndex.containsKey(feature) && sparseFeatureIndex.get(feature).values().contains(index)) {
            // we can map this index to an interpretable string, so we do
            for (String s : sparseFeatureIndex.get(feature).keySet()) {
                if (sparseFeatureIndex.get(feature).get(s) == index) {
                    sb.append(s);
                    break;
                }
            }
        }
        else {
            // we can't map this to a useful string, so we default to the number
            sb.append(Integer.toString(index));
        }
        sb.append(": ");
        sb.append(Double.toString(vector.getValueAt(featureToIndex.get(feature), index)));
        sb.append("\n");
    }
}
