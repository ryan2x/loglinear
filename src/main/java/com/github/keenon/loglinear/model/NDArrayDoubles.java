package com.github.keenon.loglinear.model;

import com.github.keenon.loglinear.NDArrayDoublesProto;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Iterator;

/**
 * Created by keenon on 9/12/15.
 * <p>
 * Holds and provides access to an N-dimensional array.
 * <p>
 * Yes, generics will lead to unfortunate boxing and unboxing in the TableFactor case, we'll handle that if it becomes a
 * problem.
 */
public class NDArrayDoubles implements Iterable<int[]> {
  // public data
  protected int[] dimensions;

  // OPTIMIZATION:
  // in normal NDArray this is private, but to allow for optimizations we actually leave it as protected
  protected double[] values;

  /**
   * Constructor takes a list of neighbor variables to use for this factor. This must not change after construction,
   * and the number of states of those variables must also not change.
   *
   * @param dimensions list of neighbor variables assignment range sizes
   */
  public NDArrayDoubles(int[] dimensions) {
    for (int size : dimensions) {
      assert (size > 0);
    }
    this.dimensions = dimensions;
    values = new double[combinatorialNeighborStatesCount()];
  }

  private NDArrayDoubles(int[] dimensions, double[] values) {
    this.dimensions = dimensions;
    this.values = values;
  }

  /**
   * This is to enable the partially observed constructor for TableFactor. It's an ugly break of modularity, but seems
   * to be necessary if we want to keep the constructor for TableFactor with partial observations relatively simple.
   */
  protected NDArrayDoubles() {
  }


  /**
   * Copy this array.
   */
  public NDArrayDoubles deepCopy() {
    int[] dimensions = new int[this.dimensions.length];
    System.arraycopy(this.dimensions, 0, dimensions, 0, this.dimensions.length);
    double[] values = new double[this.values.length];
    System.arraycopy(this.values, 0, values, 0, this.values.length);
    return new NDArrayDoubles(dimensions, values);
  }

  /**
   * Set a single value in the factor table.
   *
   * @param assignment a list of variable settings, in the same order as the neighbors array of the factor
   * @param value      the value to put into the factor table
   */
  public void setAssignmentValue(int[] assignment, double value) {
    assert !Double.isNaN(value);
    values[getTableAccessOffset(assignment)] = value;
  }

  /**
   * Retrieve a single value for an assignment.
   *
   * @param assignment a list of variable settings, in the same order as the neighbors array of the factor
   * @return the value for the given assignment. Can be null if not been set yet.
   */
  public double getAssignmentValue(int[] assignment) {
    return values[getTableAccessOffset(assignment)];
  }

  /**
   * @return the size array of the neighbors of the feature factor, passed by value to ensure immutability.
   */
  public int[] getDimensions() {
    return dimensions.clone();
  }

  /**
   * WARNING: This is pass by reference to avoid massive GC overload during heavy iterations, and because the standard
   * use case is to use the assignments array as an accessor. Please, clone if you save a copy, otherwise the array
   * will mutate underneath you.
   *
   * @return an iterator over all possible assignments to this factor
   */
  @Override
  public Iterator<int[]> iterator() {
    return new Iterator<int[]>() {
      Iterator<int[]> unsafe = fastPassByReferenceIterator();

      @Override
      public boolean hasNext() {
        return unsafe.hasNext();
      }

      @Override
      public int[] next() {
        return unsafe.next().clone();
      }
    };
  }

  /**
   * This is its own function because people will inevitably attempt this optimization of not cloning the array we
   * hand to the iterator, to save on GC, and it should not be default behavior. If you know what you're doing, then
   * this may be the iterator for you.
   *
   * @return an iterator that will mutate the value it returns to you, so you must clone if you want to keep a copy
   */
  public Iterator<int[]> fastPassByReferenceIterator() {
    final int[] assignments = new int[dimensions.length];
    if (dimensions.length > 0) assignments[0] = -1;

    return new Iterator<int[]>() {
      @Override
      public boolean hasNext() {
        for (int i = 0; i < assignments.length; i++) {
          if (assignments[i] < dimensions[i] - 1) return true;
        }
        return false;
      }

      @Override
      public int[] next() {
        // Add one to the first position
        assignments[0]++;
        // Carry any resulting overflow all the way to the end.
        for (int i = 0; i < assignments.length; i++) {
          if (assignments[i] >= dimensions[i]) {
            assignments[i] = 0;
            if (i < assignments.length - 1) {
              assignments[i + 1]++;
            }
          } else {
            break;
          }
        }
        return assignments;
      }
    };
  }

  /**
   * Does a deep comparison, using equality with tolerance checks against the vector table of values.
   *
   * @param other     the factor to compare to
   * @param tolerance the tolerance to accept in differences
   * @return whether the two factors are within tolerance of one another
   */
  public boolean valueEquals(NDArrayDoubles other, double tolerance) {
    if (!Arrays.equals(dimensions, other.dimensions)) return false;
    if (dimensions.length != other.dimensions.length) return false;
    for (int i = 0; i < dimensions.length; i++) {
      if (Math.abs(dimensions[i] - other.dimensions[i]) > tolerance) return false;
    }
    return true;
  }

  /**
   * @return the total number of states this factor must represent to include all neighbors.
   */
  public int combinatorialNeighborStatesCount() {
    int c = 1;
    for (int n : dimensions) {
      c *= n;
    }
    return c;
  }

  /**
   * Convenience function to write this factor directly to a stream, encoded as proto. Reversible with readFromStream.
   *
   * @param stream the stream to write to. does not flush automatically
   * @throws IOException passed through from the stream
   */
  public void writeToStream(OutputStream stream) throws IOException {
    getProtoBuilder().build().writeTo(stream);
  }

  /**
   * Convenience function to read a factor (assumed serialized with proto) directly from a stream.
   *
   * @param stream the stream to be read from
   * @return a new in-memory feature factor
   * @throws IOException passed through from the stream
   */
  public static NDArrayDoubles readFromStream(InputStream stream) throws IOException {
    return readFromProto(NDArrayDoublesProto.NDArrayDoubles.parseFrom(stream));
  }

  /**
   * Returns the proto builder object for this feature factor. Recursively constructs protos for all the concat
   * vectors in factorTable.
   *
   * @return proto Builder object
   */
  public NDArrayDoublesProto.NDArrayDoubles.Builder getProtoBuilder() {
    NDArrayDoublesProto.NDArrayDoubles.Builder b = NDArrayDoublesProto.NDArrayDoubles.newBuilder();
    for (int n : getDimensions()) {
      b.addDimensionSize(n);
    }
    for (double value : values) {
      b.addValues(value);
    }
    return b;
  }

  /**
   * Creates a new in-memory feature factor from a proto serialization,
   *
   * @param proto the proto object to be turned into an in-memory feature factor
   * @return an in-memory feature factor, complete with in-memory concat vectors
   */
  public static NDArrayDoubles readFromProto(NDArrayDoublesProto.NDArrayDoubles proto) {
    int[] neighborSizes = new int[proto.getDimensionSizeCount()];
    for (int i = 0; i < neighborSizes.length; i++) {
      neighborSizes[i] = proto.getDimensionSize(i);
    }
    NDArrayDoubles factor = new NDArrayDoubles(neighborSizes);
    for (int i = 0; i < proto.getValuesCount(); i++) {
      factor.values[i] = proto.getValues(i);
      assert !Double.isNaN(factor.values[i]);
    }
    return factor;
  }

  ////////////////////////////////////////////////////////////////////////////
  // PRIVATE IMPLEMENTATION
  ////////////////////////////////////////////////////////////////////////////

  /**
   * Compute the distance into the one dimensional factorTable array that corresponds to a setting of all the
   * neighbors of the factor.
   *
   * @param assignment assignment indices, in same order as neighbors array
   * @return the offset index
   */
  private int getTableAccessOffset(int[] assignment) {
    assert (assignment.length == dimensions.length);
    int offset = 0;
    for (int i = 0; i < assignment.length; i++) {
      assert (assignment[i] < dimensions[i]);
      offset = (offset * dimensions[i]) + assignment[i];
    }
    return offset;
  }
}
