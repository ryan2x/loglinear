package com.github.keenon.loglinear.model;

import com.github.keenon.loglinear.ConcatVectorNamespaceProto;
import com.github.keenon.loglinear.ConcatVectorProto;

import java.io.*;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by keenon on 10/20/15.
 *
 * This is a wrapper function to keep a namespace of namespace of recognized features, so that building a set of
 * ConcatVectors for featurizing a model is easier and more intuitive. It's actually quite simple, and threadsafe.
 */
public class ConcatVectorNamespace implements Serializable {
    /** A serialversionuid so we can save this robustly */
    private static final long serialVersionUID = 42;

    // This is the name of a feature that we expect all weight vectors to set to 1.0
    static final String ALWAYS_ONE_FEATURE = "__lense__.ALWAYS_ONE";

    final Map<String, Integer> featureToIndex = new HashMap<>();
    final Map<String, Map<String,Integer>> sparseFeatureIndex = new HashMap<>();
    final Map<String, Map<Integer,String>> reverseSparseFeatureIndex = new HashMap<>();

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
                reverseSparseFeatureIndex.put(featureName, new HashMap<>());
            }
        }
        final Map<String,Integer> sparseIndex = sparseFeatureIndex.get(featureName);
        final Map<Integer,String> reverseSparseIndex = reverseSparseFeatureIndex.get(featureName);
        synchronized (sparseIndex) {
            if (!sparseIndex.containsKey(index)) {
                reverseSparseIndex.put(sparseIndex.size(), index);
                sparseIndex.put(index, sparseIndex.size());
            }
            return sparseIndex.get(index);
        }
    }

    /**
     * Sets the special "always one" feature slot to something. For weight vectors, this should always be set to 1.0.
     * For everyone else, this can be set to whatever people want.
     *
     * @param vector the vector we'd like to set
     * @param value the value we'd like to set it to
     */
    public void setAlwaysOneFeature(ConcatVector vector, double value) {
        setDenseFeature(vector, ALWAYS_ONE_FEATURE, new double[]{value});
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
     * This adds a sparse set feature to a vector, setting the appropriate components of the given vector to the passed
     * in value.
     * @param vector the vector
     * @param featureName the feature whose value to set
     * @param sparseFeatures the indices we wish to set, and their values
     */
    public void setSparseFeature(ConcatVector vector, String featureName, Map<String,Double> sparseFeatures) {
        int[] indices = new int[sparseFeatures.size()];
        double[] values = new double[sparseFeatures.size()];
        int offset = 0;
        for (String index : sparseFeatures.keySet()) {
            indices[offset] = ensureSparseFeature(featureName, index);
            values[offset] = sparseFeatures.get(index);
            offset++;
        }
        vector.setSparseComponent(ensureFeature(featureName), indices, values);
    }

    /**
     * This adds a sparse set feature to a vector, setting the appropriate components of the given vector to the passed
     * in value.
     * @param vector the vector
     * @param featureName the feature whose value to set
     * @param sparseFeatures the indices we wish to set, whose values will all be set to 1.0
     */
    public void setSparseFeature(ConcatVector vector, String featureName, Collection<String> sparseFeatures) {
        int[] indices = new int[sparseFeatures.size()];
        double[] values = new double[sparseFeatures.size()];
        int offset = 0;
        for (String index : sparseFeatures) {
            indices[offset] = ensureSparseFeature(featureName, index);
            values[offset] = 1.0;
            offset++;
        }
        vector.setSparseComponent(ensureFeature(featureName), indices, values);
    }

    /**
     * Writes the protobuf version of this vector to a stream. reversible with readFromStream().
     *
     * @param stream the output stream to write to
     * @throws IOException passed through from the stream
     */
    public void writeToStream(OutputStream stream) throws IOException {
        getProtoBuilder().build().writeDelimitedTo(stream);
    }

    /**
     * Static function to deserialize a concat vector from an input stream.
     *
     * @param stream the stream to read from, assuming protobuf encoding
     * @return a new concat vector
     * @throws IOException passed through from the stream
     */
    public static ConcatVectorNamespace readFromStream(InputStream stream) throws IOException {
        return readFromProto(ConcatVectorNamespaceProto.ConcatVectorNamespace.parseDelimitedFrom(stream));
    }

    /**
     * @return a Builder for proto serialization
     */
    public ConcatVectorNamespaceProto.ConcatVectorNamespace.Builder getProtoBuilder() {
        ConcatVectorNamespaceProto.ConcatVectorNamespace.Builder m = ConcatVectorNamespaceProto.ConcatVectorNamespace.newBuilder();

        // Add the outer layer features
        for (String feature : featureToIndex.keySet()) {
            ConcatVectorNamespaceProto.ConcatVectorNamespace.FeatureToIndexComponent.Builder component = ConcatVectorNamespaceProto.ConcatVectorNamespace.FeatureToIndexComponent.newBuilder();

            component.setKey(feature);
            component.setData(featureToIndex.get(feature));

            m.addFeatureToIndex(component);
        }

        for (String feature : sparseFeatureIndex.keySet()) {
            ConcatVectorNamespaceProto.ConcatVectorNamespace.SparseFeatureIndex.Builder sparseFeature = ConcatVectorNamespaceProto.ConcatVectorNamespace.SparseFeatureIndex.newBuilder();

            sparseFeature.setKey(feature);
            for (String sparseFeatureName : sparseFeatureIndex.get(feature).keySet()) {
                ConcatVectorNamespaceProto.ConcatVectorNamespace.FeatureToIndexComponent.Builder component = ConcatVectorNamespaceProto.ConcatVectorNamespace.FeatureToIndexComponent.newBuilder();
                component.setKey(sparseFeatureName);
                component.setData(sparseFeatureIndex.get(feature).get(sparseFeatureName));
                sparseFeature.addFeatureToIndex(component);
            }

            m.addSparseFeatureIndex(sparseFeature);
        }

        return m;
    }

    /**
     * Recreates an in-memory concat vector object from a Proto serialization.
     *
     * @param m the concat vector proto
     * @return an in-memory concat vector object
     */
    public static ConcatVectorNamespace readFromProto(ConcatVectorNamespaceProto.ConcatVectorNamespace m) {
        ConcatVectorNamespace namespace = new ConcatVectorNamespace();

        for (ConcatVectorNamespaceProto.ConcatVectorNamespace.FeatureToIndexComponent component : m.getFeatureToIndexList()) {
            namespace.featureToIndex.put(component.getKey(), component.getData());
        }

        for (ConcatVectorNamespaceProto.ConcatVectorNamespace.SparseFeatureIndex sparseFeature : m.getSparseFeatureIndexList()) {
            String key = sparseFeature.getKey();
            Map<String, Integer> sparseMap = new HashMap<>();
            for (ConcatVectorNamespaceProto.ConcatVectorNamespace.FeatureToIndexComponent component : sparseFeature.getFeatureToIndexList()) {
                sparseMap.put(component.getKey(), component.getData());
            }
            namespace.sparseFeatureIndex.put(key, sparseMap);
        }

        return namespace;
    }

    /**
     * This prints out a ConcatVector by mapping to the namespace, to make debugging learning algorithms easier.
     *
     * @param vector the vector to print
     * @param bw the output stream to write to
     */
    public void debugVector(ConcatVector vector, BufferedWriter bw) throws IOException {
        for (String key : featureToIndex.keySet()) {
            bw.write(key);
            bw.write(":\n");
            int i = featureToIndex.get(key);
            if (vector.isComponentSparse(i)) {
                int[] indices = vector.getSparseIndices(i);
                for (int j : indices) {
                    debugFeatureValue(key, j, vector, bw);
                }
            }
            else {
                double[] arr = vector.getDenseComponent(i);
                for (int j = 0; j < arr.length; j++) {
                    debugFeatureValue(key, j, vector, bw);
                }
            }
        }
    }

    /**
     * This writes a feature's individual value, using the human readable name if possible, to a StringBuilder
     */
    private void debugFeatureValue(String feature, int index, ConcatVector vector, BufferedWriter bw) throws IOException {
        bw.write("\t");
        if (sparseFeatureIndex.containsKey(feature) && sparseFeatureIndex.get(feature).values().contains(index)) {
            // we can map this index to an interpretable string, so we do
            bw.write(reverseSparseFeatureIndex.get(feature).get(index));
        }
        else {
            // we can't map this to a useful string, so we default to the number
            bw.write(Integer.toString(index));
        }
        bw.write(": ");
        bw.write(Double.toString(vector.getValueAt(featureToIndex.get(feature), index)));
        bw.write("\n");
    }
}
