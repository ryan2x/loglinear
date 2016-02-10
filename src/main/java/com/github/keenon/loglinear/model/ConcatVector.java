package com.github.keenon.loglinear.model;

import com.github.keenon.loglinear.ConcatVectorProto;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.function.Function;

/**
 * Created by keenon on 12/7/14.
 *
 * Implements a concat vector using an array of arrays, with all its attending resizing efficiencies, and double-pointer
 * inefficiencies. Benchmarking from MinimalML (where I adapted this design from) shows that this is the most efficient
 * of several strategies that can be used to implement this.
 *
 * What is a ConcatVector? Why do I need it?
 *
 * In short, you want this for online learning, where you may not know all your sparse features' sizes at initialization.
 * A concat vector is a vector that behaves like a concatenation of smaller component vectors when you want a dot product.
 * However, it never physically concatenates anything, it just dot products each component, and takes the sum. That way,
 * if you need to expand a component during online learning, it's no problem. As an auxiliary benefit, you can specify
 * sparse and dense components, greatly speeding up dot product calculation when you have lots of sparse features.
 */
public class ConcatVector {
    double[][] pointers;
    boolean[] sparse;
    boolean[] copyOnWrite;

    /**
     * Constructor that initializes space for this concat vector. Don't worry, it can resize individual elements as
     * necessary but it's most efficient if you get this right at construction.
     *
     * @param numComponents The number of components (usually number of features) to allocate for.
     */
    public ConcatVector(int numComponents) {
        pointers = new double[numComponents][];
        sparse = new boolean[numComponents];
        copyOnWrite = new boolean[numComponents];
    }

    /**
     * Clone a concat vector constructor. Marks both vectors as copyOnWrite, but makes no immediate copies.
     * @param clone the concat vector to clone.
     */
    private ConcatVector(ConcatVector clone) {
        pointers = new double[clone.pointers.length][];
        copyOnWrite = new boolean[clone.pointers.length];
        for (int i = 0; i < clone.pointers.length; i++) {
            if (clone.pointers[i] == null) continue;
            pointers[i] = clone.pointers[i];
            copyOnWrite[i] = true;
            clone.copyOnWrite[i] = true;
        }
        sparse = new boolean[clone.pointers.length];
        if (clone.pointers.length > 0) {
            System.arraycopy(clone.sparse, 0, sparse, 0, clone.pointers.length);
        }
    }

    /**
     * Creates a ConcatVector whose dimensions are the same as this one for all dense components, but is otherwise
     * completely empty. This is useful to prevent resizing during optimizations where we're adding lots of sparse
     * vectors.
     *
     * @return an empty vector suitable for use as a gradient
     */
    public ConcatVector newEmptyClone() {
        ConcatVector clone = new ConcatVector(getNumberOfComponents());
        for (int i = 0; i < pointers.length; i++) {
            if (pointers[i] != null && !sparse[i]) {
                clone.pointers[i] = new double[pointers[i].length];
                clone.sparse[i] = false;
            }
        }
        return clone;
    }

    /**
     * Sets a single component of the concat vector value as a dense vector. This will make a copy of you values array,
     * so you're free to continue mutating it.
     *
     * @param component the index of the component to set
     * @param values the array of dense values to put into the component
     */
    public void setDenseComponent(int component, double[] values) {
        if (component >= pointers.length) {
            increaseSizeTo(component+1);
        }
        pointers[component] = values;
        sparse[component] = false;
        copyOnWrite[component] = true;
    }

    /**
     * Sets a single component of the concat vector value as a sparse, one hot value.
     *
     * @param component the index of the component to set
     * @param index the index of the vector to one-hot
     * @param value the value of that index
     */
    public void setSparseComponent(int component, int index, double value) {
        if (component >= pointers.length) {
            increaseSizeTo(component+1);
        }
        double[] sparseInfo = new double[2];
        sparseInfo[0] = index;
        sparseInfo[1] = value;
        pointers[component] = sparseInfo;
        sparse[component] = true;
        copyOnWrite[component] = false;
    }

    /**
     * Sets a component to a set of sparse indices, each with a value.
     *
     * @param component the index of the component to set
     * @param indices the indices of the vector to give values to
     * @param values their values
     */
    public void setSparseComponent(int component, int[] indices, double[] values) {
        if (component >= pointers.length) {
            increaseSizeTo(component+1);
        }

        assert(indices.length == values.length);

        if (indices.length == 0) {
            pointers[component] = new double[2];
            sparse[component] = true;
            copyOnWrite[component] = false;
        }
        else {
            double[] sparseInfo = new double[indices.length * 2];
            for (int i = 0; i < indices.length; i++) {
                sparseInfo[i * 2] = indices[i];
                sparseInfo[(i * 2) + 1] = values[i];
            }
            pointers[component] = sparseInfo;
            sparse[component] = true;
            copyOnWrite[component] = false;
        }
    }

    /**
     * This function assumes both vectors are infinitely padded with 0s, so it won't complain if there's a dim mismatch.
     * There are no side effects.
     *
     * @param other the MV to dot product with
     * @return the dot product of this and other
     */
    public double dotProduct(ConcatVector other) {
        if (loadedNative) {
            return dotProductNative(other);
        }
        else {
            double sum = 0.0f;
            for (int i = 0; i < Math.min(pointers.length, other.pointers.length); i++) {
                if (pointers[i] == null || other.pointers[i] == null) continue;
                if (sparse[i] && other.sparse[i]) {
                    outer: for (int j = 0; j < pointers[i].length/2; j++) {
                        int sparseIndex = (int)pointers[i][j*2];
                        for (int k = 0; k < other.pointers[i].length/2; k++) {
                            int otherSparseIndex = (int) other.pointers[i][k*2];
                            if (sparseIndex == otherSparseIndex) {
                                sum += pointers[i][(j*2)+1]*other.pointers[i][(k*2)+1];
                                continue outer;
                            }
                        }
                    }
                } else if (sparse[i] && !other.sparse[i]) {
                    for (int j = 0; j < pointers[i].length/2; j++) {
                        int sparseIndex = (int) pointers[i][j*2];
                        if (sparseIndex >= 0 && sparseIndex < other.pointers[i].length) {
                            sum += other.pointers[i][sparseIndex] * pointers[i][(j*2)+1];
                        }
                    }
                } else if (!sparse[i] && other.sparse[i]) {
                    for (int j = 0; j < other.pointers[i].length/2; j++) {
                        int sparseIndex = (int) other.pointers[i][j * 2];
                        if (sparseIndex >= 0 && sparseIndex < pointers[i].length) {
                            sum += pointers[i][sparseIndex] * other.pointers[i][(j*2)+1];
                        }
                    }
                } else {
                    for (int j = 0; j < Math.min(pointers[i].length, other.pointers[i].length); j++) {
                        sum += pointers[i][j] * other.pointers[i][j];
                    }
                }
            }
            return sum;
        }
    }

    /**
     * @return a clone of this concat vector, with deep copies of datastructures
     */
    public ConcatVector deepClone() {
        return new ConcatVector(this);
    }

    /**
     * This will add the vector "other" to this vector, scaling other by multiple. In algebra,
     *
     * this = this + (other * multiple)
     *
     * The function assumes that both vectors are padded infinitely with 0s, so will scale this vector by adding components
     * and changing component sizes (dense to bigger dense) and shapes (sparse to dense) in order to accommodate the result.
     *
     * @param other the vector to add to this one
     * @param multiple the multiple to use
     */
    public void addVectorInPlace(ConcatVector other, double multiple) {
        // Resize if necessary
        if (pointers == null) {
            pointers = new double[other.pointers.length][];
            sparse = new boolean[other.pointers.length];
            copyOnWrite = new boolean[other.pointers.length];
        } else if (pointers.length < other.pointers.length) {
            increaseSizeTo(other.pointers.length);
        }

        // Do the addition piece by piece

        for (int i = 0; i < other.pointers.length; i++) {
            // If the other vector has no segment here, then skip
            if (other.pointers[i] == null) continue;
            // If we previously had no element here, fill it in accordingly
            if (pointers[i] == null || pointers[i].length == 0) {
                sparse[i] = other.sparse[i];
                // If the multiple is one, just follow the copying procedure
                if (multiple == 1.0) {
                    pointers[i] = other.pointers[i];
                    copyOnWrite[i] = true;
                    other.copyOnWrite[i] = true;
                }
                // Otherwise do the standard thing
                else {
                    if (other.sparse[i]) {
                        pointers[i] = new double[other.pointers[i].length];
                        copyOnWrite[i] = false;
                        for (int j = 0; j < other.pointers[i].length/2; j++) {
                            pointers[i][j*2] = other.pointers[i][j*2];
                            pointers[i][(j*2)+1] = other.pointers[i][(j*2)+1] * multiple;
                        }
                    } else {
                        pointers[i] = new double[other.pointers[i].length];
                        copyOnWrite[i] = false;
                        for (int j = 0; j < other.pointers[i].length; j++) {
                            pointers[i][j] = other.pointers[i][j] * multiple;
                        }
                    }
                }
            }
            // Handle rescaling on a component-by-component basis
            else if (sparse[i] && !other.sparse[i]) {
                int maxSparseIndex = -1;
                for (int j = 0; j < pointers[i].length/2; j++) {
                    int sparseIndex = (int) pointers[i][j*2];
                    if (sparseIndex > maxSparseIndex) maxSparseIndex = sparseIndex;
                }
                // Convert to a dense vector
                double[] newPointers = new double[Math.max(maxSparseIndex + 1, other.pointers[i].length)];
                for (int j = 0; j < pointers[i].length/2; j++) {
                    int sparseIndex = (int) pointers[i][j*2];
                    if (sparseIndex >= 0) newPointers[sparseIndex] = pointers[i][(j*2)+1];
                }
                // Add the other vector's dense values, multiplied by the scalar
                for (int j = 0; j < other.pointers[i].length; j++) {
                    newPointers[j] += other.pointers[i][j] * multiple;
                }
                // Update
                sparse[i] = false;
                copyOnWrite[i] = false;
                pointers[i] = newPointers;
            } else if (sparse[i] && other.sparse[i]) {

                // Figure out how big the vector would be if it were dense

                int maxSparseIndex = 0;
                for (int j = 0; j < pointers[i].length/2; j++) {
                    int sparseIndex = (int) pointers[i][j*2];
                    if (sparseIndex > maxSparseIndex) maxSparseIndex = sparseIndex;
                }
                for (int j = 0; j < other.pointers[i].length/2; j++) {
                    int sparseIndex = (int) other.pointers[i][j*2];
                    if (sparseIndex > maxSparseIndex) maxSparseIndex = sparseIndex;
                }

                // Figure out (an upper bound on) how big the vector would be if it remained sparse

                int numEntries = pointers[i].length/2 + other.pointers[i].length/2;

                // Only switch over to dense if the physical double array will be smaller

                if (numEntries*2 > maxSparseIndex) {
                    sparse[i] = false;
                    double[] newPointers = new double[maxSparseIndex + 1];
                    copyOnWrite[i] = false;
                    for (int j = 0; j < pointers[i].length/2; j++) {
                        int sparseIndex = (int) pointers[i][j*2];
                        if (sparseIndex >= 0) newPointers[sparseIndex] = pointers[i][(j*2)+1];
                    }
                    for (int j = 0; j < other.pointers[i].length/2; j++) {
                        int sparseIndex = (int) other.pointers[i][j*2];
                        if (sparseIndex >= 0) newPointers[sparseIndex] += other.pointers[i][(j*2)+1] * multiple;
                    }
                    pointers[i] = newPointers;
                }

                // Otherwise compose a joint sparse array

                else {
                    int duplicates = 0;
                    outer: for (int j = 0; j < pointers[i].length/2; j++) {
                        for (int k = 0; k < other.pointers[i].length/2; k++) {
                            if (pointers[i][j*2] == other.pointers[i][k*2]) {
                                duplicates++;
                                continue outer;
                            }
                        }
                    }

                    double[] newPointers = new double[(numEntries-duplicates)*2];
                    int usedNewPointers = 0;

                    for (int j = 0; j < pointers[i].length/2; j++) {
                        newPointers[j*2] = pointers[i][j*2];
                        newPointers[(j*2)+1] = pointers[i][(j*2)+1];
                        usedNewPointers = j+1;
                    }
                    outer: for (int j = 0; j < other.pointers[i].length/2; j++) {
                        int otherSparseIndex = (int) other.pointers[i][j*2];
                        for (int k = 0; k < usedNewPointers; k++) {
                            int sparseIndex = (int) newPointers[k*2];
                            if (otherSparseIndex == sparseIndex) {
                                newPointers[(k*2)+1] += other.pointers[i][(j*2)+1]*multiple;
                                continue outer;
                            }
                        }
                        // If this isn't true we somehow messed up calculating the number of sparse entries needed
                        assert(usedNewPointers < newPointers.length);
                        newPointers[(usedNewPointers*2)] = other.pointers[i][j*2];
                        newPointers[(usedNewPointers*2)+1] = other.pointers[i][(j*2)+1]*multiple;
                        usedNewPointers++;
                    }

                    copyOnWrite[i] = false;
                    sparse[i] = true;
                    pointers[i] = newPointers;
                }
            } else if (!sparse[i] && other.sparse[i]) {
                int maxSparseIndex = 0;
                for (int j = 0; j < other.pointers[i].length/2; j++) {
                    int sparseIndex = (int) other.pointers[i][j*2];
                    if (sparseIndex > maxSparseIndex) maxSparseIndex = sparseIndex;
                }
                if (maxSparseIndex >= pointers[i].length) {
                    int newSize = pointers[i].length;
                    while (newSize <= maxSparseIndex) newSize *= 2;
                    double[] denseBuf = new double[newSize];
                    System.arraycopy(pointers[i], 0, denseBuf, 0, pointers[i].length);
                    copyOnWrite[i] = false;
                    pointers[i] = denseBuf;
                }
                if (copyOnWrite[i]) {
                    pointers[i] = pointers[i].clone();
                    copyOnWrite[i] = false;
                }
                for (int j = 0; j < other.pointers[i].length/2; j++) {
                    int sparseIndex = (int) other.pointers[i][j*2];
                    if (sparseIndex >= 0) pointers[i][sparseIndex] += other.pointers[i][(j*2)+1] * multiple;
                }
            } else {
                assert (!sparse[i] && !other.sparse[i]);
                if (pointers[i].length < other.pointers[i].length) {
                    double[] denseBuf = new double[other.pointers[i].length];
                    System.arraycopy(pointers[i], 0, denseBuf, 0, pointers[i].length);
                    copyOnWrite[i] = false;
                    pointers[i] = denseBuf;
                }
                if (copyOnWrite[i]) {
                    pointers[i] = pointers[i].clone();
                    copyOnWrite[i] = false;
                }
                for (int j = 0; j < other.pointers[i].length; j++) {
                    pointers[i][j] += other.pointers[i][j] * multiple;
                }
            }
        }
    }

    /**
     * This will multiply the vector "other" to this vector. It's the equivalent of the Matlab
     *
     * this = this .* other
     *
     * The function assumes that both vectors are padded infinitely with 0s, so will result in lots of 0s in this
     * vector if it is longer than 'other'.
     *
     * @param other the vector to multiply into this one
     */
    public void elementwiseProductInPlace(ConcatVector other) {
        for (int i = 0; i < pointers.length; i++) {
            if (pointers[i] == null) continue;

            if (copyOnWrite[i]) {
                copyOnWrite[i] = false;
                pointers[i] = pointers[i].clone();
            }

            if (i >= other.pointers.length || other.pointers[i] == null) {
                pointers[i] = null;
            }
            else if (sparse[i] && other.sparse[i]) {
                outer: for (int j = 0; j < pointers[i].length/2; j++) {
                    int sparseValue = (int) pointers[i][j*2];
                    for (int k = 0; k < other.pointers[i].length/2; k++) {
                        int otherSparseValue = (int) other.pointers[i][k*2];
                        if (sparseValue == otherSparseValue) {
                            pointers[i][(j*2)+1] *= other.pointers[i][(k*2)+1];
                            continue outer;
                        }
                    }
                    pointers[i][(j*2)+1] = 0;
                }
            }
            else if (sparse[i] && !other.sparse[i]) {
                for (int j = 0; j < pointers[i].length/2; j++) {
                    int sparseIndex = (int)pointers[i][j*2];
                    if (sparseIndex >= 0 && sparseIndex < other.pointers[i].length) {
                        pointers[i][(j*2)+1] *= other.pointers[i][sparseIndex];
                    }
                    else {
                        pointers[i][(j*2)+1] = 0.0f;
                    }
                }
            }
            else if (!sparse[i] && other.sparse[i]) {
                // Note: there are cases where we want this element to actually become sparse, but I can't think of any
                // in practice, so we won't bother with the complexity
                boolean[] touched = new boolean[pointers[i].length];
                for (int j = 0; j < other.pointers[i].length/2; j++) {
                    int sparseIndex = (int)other.pointers[i][j*2];
                    if (sparseIndex >= 0 && sparseIndex < pointers[i].length) {
                        pointers[i][sparseIndex] *= other.pointers[i][(j*2)+1];
                        touched[sparseIndex] = true;
                    }
                }
                for (int j = 0; j < pointers[i].length; j++) {
                    if (!touched[j]) pointers[i][j] = 0.0;
                }
            }
            else {
                for (int j = 0; j < Math.min(pointers[i].length, other.pointers[i].length); j++) {
                    pointers[i][j] *= other.pointers[i][j];
                }
                for (int j = other.pointers[i].length; j < pointers[i].length; j++) {
                    pointers[i][j] = 0.0f;
                }
            }
        }
    }

    /**
     * Apply a function to every element of every component of this vector, and replace with the result.
     *
     * @param fn the function to apply to every element of every component.
     */
    public void mapInPlace(Function<Double, Double> fn) {
        for (int i = 0; i < pointers.length; i++) {
            if (pointers[i] == null) continue;

            if (copyOnWrite[i]) {
                copyOnWrite[i] = false;
                pointers[i] = pointers[i].clone();
            }

            if (sparse[i]) {
                for (int j = 0; j < pointers[i].length/2; j++) {
                    pointers[i][(j*2)+1] = fn.apply(pointers[i][(j*2)+1]);
                }
            }
            else {
                for (int j = 0; j < pointers[i].length; j++) {
                    pointers[i][j] = fn.apply(pointers[i][j]);
                }
            }
        }
    }

    /**
     * @return the number of concatenated vectors that compose this ConcatVector
     */
    public int getNumberOfComponents() {
        return pointers.length;
    }

    /**
     * @param i the index of the component to check
     * @return whether component i is sparse or not
     */
    public boolean isComponentSparse(int i) {
        return sparse[i];
    }

    /**
     * This function will throw an assert if the component you're requesting isn't dense
     *
     * @param i the index of the component to look at
     * @return the dense array composing that component
     */
    public double[] getDenseComponent(int i) {
        assert(!sparse[i]);
        // This will save the special case code down the line, so is worth the tiny object creation
        if (pointers[i] == null) return new double[0];
        return pointers[i];
    }

    /**
     * This assumes infinite padding with 0s. It will return you 0 if you're OOB (use getSegmentSizes() to check, if
     * that's undesirable behavior). Otherwise it will return you the correct value.
     *
     * @param component the index of the component to retrieve a value from
     * @param offset the offset within that component
     * @return the value retrieved, of 0 if OOB
     */
    public double getValueAt(int component, int offset) {
        if (component < pointers.length) {
            if (pointers[component] == null) return 0;
            else if (sparse[component]) {
                for (int j = 0; j < pointers[component].length/2; j++) {
                    int sparseIndex = (int)pointers[component][j*2];
                    if (sparseIndex == offset) return pointers[component][(j*2)+1];
                }
            }
            else {
                if (offset < pointers[component].length) {
                    return pointers[component][offset];
                }
            }
        }
        return 0;
    }

    /**
     * Gets you the index of one hot in a component, assuming it is sparse. Throws an assert if it isn't.
     *
     * @param component the index of the sparse component.
     * @return the index of the one-hot value within that sparse component.
     */
    public int getSparseIndex(int component) {
        assert(sparse[component]);
        return (int)pointers[component][0];
    }

    /**
     * Gets you the indices of multi hot in a component, assuming it is sparse. Throws an assert if it isn't.
     *
     * @param component the index of the sparse component.
     * @return the index of the one-hot value within that sparse component.
     */
    public int[] getSparseIndices(int component) {
        assert(sparse[component]);
        int[] indices = new int[pointers[component].length/2];
        for (int i = 0; i < pointers[component].length/2; i++) {
            indices[i] = (int)pointers[component][i*2];
        }
        return indices;
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
    public static ConcatVector readFromStream(InputStream stream) throws IOException {
        return readFromProto(ConcatVectorProto.ConcatVector.parseDelimitedFrom(stream));
    }

    /**
     * @return a Builder for proto serialization
     */
    public ConcatVectorProto.ConcatVector.Builder getProtoBuilder() {
        ConcatVectorProto.ConcatVector.Builder m = ConcatVectorProto.ConcatVector.newBuilder();
        for (int i = 0; i < pointers.length; i++) {
            ConcatVectorProto.ConcatVector.Component.Builder c = ConcatVectorProto.ConcatVector.Component.newBuilder();
            c.setSparse(sparse[i]);
            // We want to keep the data array size 0 if the pointers for this component is null
            if (pointers[i] != null) {
                for (int j = 0; j < pointers[i].length; j++) {
                    c.addData(pointers[i][j]);
                }
            }
            m.addComponent(c);
        }
        return m;
    }

    /**
     * Recreates an in-memory concat vector object from a Proto serialization.
     *
     * @param m the concat vector proto
     * @return an in-memory concat vector object
     */
    public static ConcatVector readFromProto(ConcatVectorProto.ConcatVector m) {
        int components = m.getComponentCount();

        ConcatVector vec = new ConcatVector();
        vec.pointers = new double[components][];
        vec.sparse = new boolean[components];
        vec.copyOnWrite = new boolean[components];
        for (int i = 0; i < components; i++) {
            ConcatVectorProto.ConcatVector.Component c = m.getComponent(i);
            vec.sparse[i] = c.getSparse();
            int dataSize = c.getDataCount();
            vec.pointers[i] = new double[dataSize];
            for (int j = 0; j < dataSize; j++) {
                vec.pointers[i][j] = c.getData(j);
            }
        }

        return vec;
    }

    /**
     * Compares two concat vectors by value. This means that we're 0 padding, so a dense and sparse component might
     * both be considered the same, if the dense array reflects the same value as the sparse array. This is pretty much
     * only useful for testing. Since it's primarily for testing, we went with the slower, more obviously correct design.
     *
     * @param other the vector we're comparing to
     * @param tolerance the amount any pair of values can differ before we say the two vectors are different.
     * @return whether the two vectors are the same
     */
    public boolean valueEquals(ConcatVector other, double tolerance) {
        for (int i = 0; i < Math.max(pointers.length, other.pointers.length); i++) {
            int size = 0;
            // Find the maximum non-zero element in this component
            if (i < pointers.length && i < other.pointers.length && pointers[i] == null && other.pointers[i] == null) {
                size = 0;
            }
            else if (i >= pointers.length || (i < pointers.length && pointers[i] == null)) {
                if (i >= other.pointers.length) {
                    size = 0;
                }
                else if (other.sparse[i]) {
                    size = other.getSparseIndex(i) + 1;
                }
                else {
                    size = other.pointers[i].length;
                }
            }
            else if (i >= other.pointers.length || (i < other.pointers.length && other.pointers[i] == null)) {
                if (i >= pointers.length) {
                    size = 0;
                }
                else if (sparse[i]) {
                    size = getSparseIndex(i) + 1;
                }
                else {
                    size = pointers[i].length;
                }
            }
            else {
                if (sparse[i] && getSparseIndex(i) >= size) size = getSparseIndex(i) + 1;
                else if (!sparse[i] && pointers[i].length > size) size = pointers[i].length;
                if (other.sparse[i] && other.getSparseIndex(i) >= size) size = other.getSparseIndex(i) + 1;
                else if (!other.sparse[i] && other.pointers[i].length > size) size = other.pointers[i].length;
            }

            for (int j = 0; j < size; j++) {
                if (Math.abs(getValueAt(i,j) - other.getValueAt(i,j)) > tolerance) return false;
            }
        }
        return true;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < pointers.length; i++) {
            sb.append(" ..");
            if (pointers[i] == null) {
                sb.append("0=0.0");
            }
            else if (sparse[i]) {
                for (int j = 0; j < pointers[i].length/2; j++) {
                    if (j > 0) sb.append(",");
                    sb.append((int)pointers[i][j*2]).append("=").append(pointers[i][(j*2)+1]);
                }
            }
            else {
                for (int j = 0; j < pointers[i].length; j++) {
                    sb.append(pointers[i][j]);
                    if (j != pointers[i].length-1) sb.append(" ");
                }
            }
            sb.append("..");
        }
        sb.append(" ]");
        return sb.toString();
    }

    ////////////////////////////////////////////////////////////////////////////
    // PRIVATE IMPLEMENTATION
    ////////////////////////////////////////////////////////////////////////////

    /**
     * This increases the length of the vector, while preserving its contents
     * @param newSize the new size to increase to. Must be larger than the current size
     */
    private void increaseSizeTo(int newSize) {
        assert(newSize > pointers.length);
        double[][] pointersBuf = new double[newSize][];
        boolean[] sparseBuf = new boolean[newSize];
        boolean[] copyOnWriteBuf = new boolean[newSize];
        System.arraycopy(pointers, 0, pointersBuf, 0, pointers.length);
        System.arraycopy(sparse, 0, sparseBuf, 0, pointers.length);
        System.arraycopy(copyOnWrite, 0, copyOnWriteBuf, 0, pointers.length);
        pointers = pointersBuf;
        sparse = sparseBuf;
        copyOnWrite = copyOnWriteBuf;
    }

    static boolean loadedNative = false;

    // Right now I'm not loading the native library even if it's available, since the dot product "speedup" is actually
    // 10x slower. First need to diagnose if a speedup is possible by going through the JNI, which is unlikely.

    /*
    static {
        try {
            System.load(System.getProperty("user.dir")+"/src/main/c/libconcatvec.so");
            loadedNative = true;
        }
        catch (UnsatisfiedLinkError e) {
            log.info("Couldn't find the native acceleration library for ConcatVector");
        }
    }
    */

    private native double dotProductNative(ConcatVector other);

    /**
     * DO NOT USE. FOR SERIALIZERS ONLY.
     */
    private ConcatVector() {}
}
