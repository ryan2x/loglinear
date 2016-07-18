package com.github.keenon.loglinear.model;

import com.github.keenon.loglinear.GraphicalModelProto;
import com.github.keenon.loglinear.inference.CliqueTree;
import com.github.keenon.loglinear.learning.LogLikelihoodDifferentiableFunction;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Created by keenon on 8/7/15.
 * <p>
 * A basic graphical model representation: Factors and Variables. This should be a fairly familiar interface to anybody
 * who's taken a basic PGM course (eg https://www.coursera.org/course/pgm). The key points:
 * - Stitching together feature factors
 * - Attaching metadata to everything, so that different sections of the program can communicate in lots of unplanned
 * ways. For now, the planned meta-data is a lot of routing and status information to do with LENSE.
 * <p>
 * This is really just the data structure, and inference lives elsewhere and must use public interfaces to access these
 * models. We just provide basic utility functions here, and barely do that, because we pass through directly to maps
 * wherever appropriate.
 */
public class GraphicalModel {
  public Map<String, String> modelMetaData = new HashMap<>();
  public List<Map<String, String>> variableMetaData = new ArrayList<>();
  public Set<Factor> factors = new HashSet<>();

  /**
   * A single factor in this graphical model. ConcatVectorTable can be reused multiple times if the same graph (or different
   * ones) and this is the glue object that tells a model where the factor lives, and what it is connected to.
   */
  public abstract static class Factor {
    public int[] neigborIndices;
    public Map<String, String> metaData = new HashMap<>();

    /**
     * @return the factor meta-data, by reference
     */
    public Map<String, String> getMetaDataByReference() {
      return metaData;
    }

    /**
     * Does a deep comparison, using equality with tolerance checks against the vector table of values.
     *
     * @param other     the factor to compare to
     * @param tolerance the tolerance to accept in differences
     * @return whether the two factors are within tolerance of one another
     */
    public abstract boolean valueEquals(Factor other, double tolerance);

    public abstract GraphicalModelProto.Factor.Builder getProtoBuilder();

    public static Factor readFromProto(GraphicalModelProto.Factor proto) {
      switch (proto.getFactorType()) {
        case Vector:
          return VectorFactor.readFromProto(proto);
        case Static:
          return StaticFactor.readFromProto(proto);
      }
      throw new IllegalStateException("Have a proto factor type that doesn't exist");
    }

    /**
     * Duplicates this factor.
     *
     * @return a copy of the factor
     */
    public abstract Factor cloneFactor();

    /**
     * This gets the size of the N-dimensional data structure that backs this factor, which presumably is also the size
     * of each of the neighbors of the factor.
     *
     * @return the size of each variable this factor is adjacent to
     */
    public abstract int[] getDimensions();

    /**
     * This gets the value of a given assignment of the Factor, given a current set of weights (VectorFactor cares about
     * this, StaticFactor ignores it)
     *
     * @param assignment the assignment to the neighboring variables
     * @param weights the weight vector (can be ignored by implementations)
     * @return a value for this entry in the factor
     */
    public abstract double getAssignmentValue(int[] assignment, ConcatVector weights);

    /**
     * WARNING: DANGEROUS OPTIMIZATION - ONLY USE THIS IF YOU KNOW WHAT YOU ARE DOING
     *
     * This is a pass-by-reference iterator, where rather than create tons of int[] arrays on the heap as we iterate
     * through a number of assignments, we mutate an int[] array in place. This can lead to lots of unexpected behavior
     * for the unwary user, so don't use this unless you've thought about it carefully.
     */
    public abstract Iterator<int[]> fastPassByReferenceIterator();

    /**
     * The total number of states that this factor has
     */
    public abstract int combinatorialNeighborStatesCount();
  }

  /**
   * A single factor in this graphical model. ConcatVectorTable can be reused multiple times if the same graph (or different
   * ones) and this is the glue object that tells a model where the factor lives, and what it is connected to.
   */
  public static class VectorFactor extends Factor {
    public ConcatVectorTable featuresTable;

    /**
     * DO NOT USE. FOR SERIALIZATION ONLY.
     */
    private VectorFactor() {
    }

    /**
     * Creates a new VectorFactor which glues together ConcatVector valued features, and a list of neighbor variables.
     *
     * @param featuresTable the table of feature vectors (N-dimensional)
     * @param neighborIndices the list of N variables that are touched by this factor
     */
    public VectorFactor(ConcatVectorTable featuresTable, int[] neighborIndices) {
      this.featuresTable = featuresTable;
      assert neighborIndices != null;
      this.neigborIndices = neighborIndices;
    }

    @Override
    public GraphicalModelProto.Factor.Builder getProtoBuilder() {
      GraphicalModelProto.Factor.Builder builder = GraphicalModelProto.Factor.newBuilder();
      builder.setFactorType(GraphicalModelProto.FactorType.Vector);
      for (int neighbor : neigborIndices) {
        builder.addNeighbor(neighbor);
      }
      builder.setFeaturesTable(featuresTable.getProtoBuilder());
      builder.setMetaData(GraphicalModel.getProtoMetaDataBuilder(metaData));
      return builder;
    }

    public static Factor readFromProto(GraphicalModelProto.Factor proto) {
      VectorFactor factor = new VectorFactor();
      factor.featuresTable = ConcatVectorTable.readFromProto(proto.getFeaturesTable());
      factor.metaData = GraphicalModel.readMetaDataFromProto(proto.getMetaData());
      factor.neigborIndices = new int[proto.getNeighborCount()];
      for (int i = 0; i < factor.neigborIndices.length; i++) {
        factor.neigborIndices[i] = proto.getNeighbor(i);
      }
      return factor;
    }

    /**
     * Duplicates this factor.
     *
     * @return a copy of the factor
     */
    @Override
    public Factor cloneFactor() {
      VectorFactor clone = new VectorFactor();
      assert this.neigborIndices != null;
      clone.neigborIndices = neigborIndices.clone();
      clone.featuresTable = featuresTable.cloneTable();
      clone.metaData.putAll(metaData);
      return clone;
    }

    @Override
    public int[] getDimensions() {
      return featuresTable.getDimensions();
    }

    @Override
    public double getAssignmentValue(int[] assignment, ConcatVector weights) {
      return featuresTable.getAssignmentValue(assignment).get().dotProduct(weights);
    }

    @Override
    public Iterator<int[]> fastPassByReferenceIterator() {
      return featuresTable.fastPassByReferenceIterator();
    }

    @Override
    public int combinatorialNeighborStatesCount() {
      return featuresTable.combinatorialNeighborStatesCount();
    }

    /**
     * Does a deep comparison, using equality with tolerance checks against the vector table of values.
     *
     * @param other     the factor to compare to
     * @param tolerance the tolerance to accept in differences
     * @return whether the two factors are within tolerance of one another
     */
    @Override
    public boolean valueEquals(Factor other, double tolerance) {
      if (!(other instanceof VectorFactor)) return false;
      return Arrays.equals(neigborIndices, other.neigborIndices) &&
          metaData.equals(other.metaData) &&
          featuresTable.valueEquals(((VectorFactor)other).featuresTable, tolerance);
    }
  }

  public static class StaticFactor extends Factor {
    public NDArrayDoubles staticFeaturesTable;

    /**
     * DO NOT USE. FOR SERIALIZATION ONLY.
     */
    private StaticFactor() {
    }

    /**
     * Creates a new VectorFactor which glues together ConcatVector valued features, and a list of neighbor variables.
     *
     * @param staticFeaturesTable the list of factor values for each assignment to the neighbors
     * @param neighborIndices the list of N variables that are touched by this factor
     */
    public StaticFactor(NDArrayDoubles staticFeaturesTable, int[] neighborIndices) {
      assert neighborIndices != null;
      assert staticFeaturesTable != null;
      this.staticFeaturesTable = staticFeaturesTable;
      this.neigborIndices = neighborIndices;
    }

    @Override
    public boolean valueEquals(Factor other, double tolerance) {
      if (!(other instanceof StaticFactor)) return false;
      return Arrays.equals(neigborIndices, other.neigborIndices) &&
          metaData.equals(other.metaData) &&
          staticFeaturesTable.valueEquals(((StaticFactor)other).staticFeaturesTable, tolerance);
    }

    @Override
    public GraphicalModelProto.Factor.Builder getProtoBuilder() {
      GraphicalModelProto.Factor.Builder builder = GraphicalModelProto.Factor.newBuilder();
      builder.setFactorType(GraphicalModelProto.FactorType.Static);
      for (int neighbor : neigborIndices) {
        builder.addNeighbor(neighbor);
      }
      builder.setStaticFeaturesTable(staticFeaturesTable.getProtoBuilder());
      builder.setMetaData(GraphicalModel.getProtoMetaDataBuilder(metaData));
      return builder;
    }

    public static Factor readFromProto(GraphicalModelProto.Factor proto) {
      StaticFactor factor = new StaticFactor();
      factor.staticFeaturesTable = NDArrayDoubles.readFromProto(proto.getStaticFeaturesTable());
      factor.metaData = GraphicalModel.readMetaDataFromProto(proto.getMetaData());
      factor.neigborIndices = new int[proto.getNeighborCount()];
      for (int i = 0; i < factor.neigborIndices.length; i++) {
        factor.neigborIndices[i] = proto.getNeighbor(i);
      }
      return factor;
    }

    @Override
    public Factor cloneFactor() {
      int[] neighborIndices = new int[this.neigborIndices.length];
      System.arraycopy(this.neigborIndices, 0, neighborIndices, 0, this.neigborIndices.length);
      return new StaticFactor(staticFeaturesTable.deepCopy(), neighborIndices);
    }

    @Override
    public int[] getDimensions() {
      return staticFeaturesTable.getDimensions();
    }

    @Override
    public double getAssignmentValue(int[] assignment, ConcatVector ignored) {
      return staticFeaturesTable.getAssignmentValue(assignment);
    }

    @Override
    public Iterator<int[]> fastPassByReferenceIterator() {
      return staticFeaturesTable.fastPassByReferenceIterator();
    }

    @Override
    public int combinatorialNeighborStatesCount() {
      return staticFeaturesTable.combinatorialNeighborStatesCount();
    }
  }

  /**
   * @return a reference to the model meta-data
   */
  public Map<String, String> getModelMetaDataByReference() {
    return modelMetaData;
  }

  /**
   * Gets the metadata for a variable. Creates blank metadata if does not exists, then returns that. Pass by reference.
   *
   * @param variableIndex the variable number, 0 indexed, to retrieve
   * @return the metadata map corresponding to that variable number
   */
  public synchronized Map<String, String> getVariableMetaDataByReference(int variableIndex) {
    while (variableIndex >= variableMetaData.size()) {
      variableMetaData.add(new HashMap<>());
    }
    return variableMetaData.get(variableIndex);
  }

  /**
   * This adds a static factor to the model, which will always produce the same value regardless of the assignment of
   * weights during inference.
   *
   * @param neighborIndices the names of the variables, as indices
   * @param assignmentFeaturizer a function that maps from an assignment to the variables, represented as an array of
   *                             assignments in the same order as presented in neighborIndices, to a constant value.
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public StaticFactor addStaticFactor(int[] neighborIndices, int[] neighborDimensions, Function<int[], Double> assignmentFeaturizer) {
    NDArrayDoubles doubleArray = new NDArrayDoubles(neighborDimensions);
    for (int[] assignment : doubleArray) {
      doubleArray.setAssignmentValue(assignment, assignmentFeaturizer.apply(assignment));
    }
    return addStaticFactor(doubleArray, neighborIndices);
  }


  /**
   * This adds a static factor to the model, which will always produce the same value regardless of the assignment of
   * weights during inference.
   * This additionally normalizes the factor, so that it will maintain a normalized partition function; e.g.,
   * after a call to {@link CliqueTree#compileNormalizedModel()}.
   *
   * @param neighborIndices the names of the variables, as indices
   * @param assignmentFeaturizer a function that maps from an assignment to the variables, represented as an array of
   *                             assignments in the same order as presented in neighborIndices, to a constant value.
   *
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public StaticFactor addStaticNormalizedFactor(int[] neighborIndices, int[] neighborDimensions, Function<int[], Double> assignmentFeaturizer) {
    NDArrayDoubles doubleArray = new NDArrayDoubles(neighborDimensions);
    double localPartitionFunction = 0.0;
    for (int[] assignment : doubleArray) {
      localPartitionFunction += Math.exp(assignmentFeaturizer.apply(assignment));
    }
    for (int[] assignment : doubleArray) {
      doubleArray.setAssignmentValue(assignment, assignmentFeaturizer.apply(assignment) - Math.log(localPartitionFunction));
    }
    return addStaticFactor(doubleArray, neighborIndices);
  }


  /**
   * Add a conditional factor, as per a Bayes net.
   * Note that unlike the case with other factors, this function takes as input <b>probabilities</b> and not
   * unnormalized factor values.
   * In other words, we do not exponentiate the values coming from the assignment function into a log-linear model.
   *
   * @param antecedents The antecedents of the conditional probability. There can be many of these.
   * @param consequent The consequent of the conditional probability. There is only one of these.
   * @param antecedentDimensions The variable dimmensions of the antecedents.
   * @param consequentDimension The variable dimmension of the consequent.
   * @param conditionalProbaility The assignment function, taking as input an assignment of the antecedents,
   *                              and an assignment of the consequent, and producing as output a <b>probability</b>
   *                              (value between 0 and 1) for this assignment.
   *                              This can be an unnormallized probability, but it should not be a log probability, as you'd
   *                              expect from a Markov net factor.
   *
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public StaticFactor addStaticConditionalFactor(int[] antecedents, int consequent,
                                                 int[] antecedentDimensions, int consequentDimension,
                                                 BiFunction<int[], Integer, Double> conditionalProbaility) {
    // Create variables for the global factor
    int[] allDimensions = new int[antecedentDimensions.length + 1];
    System.arraycopy(antecedentDimensions, 0, allDimensions, 0, antecedentDimensions.length);
    allDimensions[allDimensions.length - 1] = consequentDimension;
    int[] allNeighbors = new int[antecedents.length + 1];
    System.arraycopy(antecedents, 0, allNeighbors, 0, antecedents.length);
    allNeighbors[allNeighbors.length - 1] = consequent;
    NDArrayDoubles totalFactor = new NDArrayDoubles(allDimensions);

    // Compute each conditional table
    NDArrayDoubles antecedentOnly = new NDArrayDoubles(antecedentDimensions);
    for (int[] assignment : antecedentOnly) {
      double localPartitionFunction = 0.0;
      for (int consequentValue = 0; consequentValue < consequentDimension; ++consequentValue) {
        localPartitionFunction += conditionalProbaility.apply(assignment, consequentValue);
      }
      for (int consequentValue = 0; consequentValue < consequentDimension; ++consequentValue) {
        int[] totalAssignment = new int[assignment.length + 1];
        System.arraycopy(assignment, 0, totalAssignment, 0, assignment.length);
        totalAssignment[totalAssignment.length - 1] = consequentValue;
        totalFactor.setAssignmentValue(totalAssignment, Math.log(conditionalProbaility.apply(assignment, consequentValue)) - Math.log(localPartitionFunction));
      }
    }

    // Create the joint factor
    return addStaticFactor(totalFactor, allNeighbors);
  }


  /**
   * This is the preferred way to add factors to a graphical model. Specify the neighbors, their dimensions, and a
   * function that maps from variable assignments to ConcatVector's of features, and this function will handle the
   * data flow of constructing and populating a factor matching those specifications.
   * <p>
   * IMPORTANT: assignmentFeaturizer must be REPEATABLE and NOT HAVE SIDE EFFECTS
   * This is because it is actually stored as a lazy closure until the full featurized vector is needed, and then it
   * is created, used, and discarded. It CAN BE CALLED MULTIPLE TIMES, and must always return the same value in order
   * for behavior of downstream systems to be defined.
   *
   * @param neighborIndices      the names of the variables, as indices
   * @param neighborDimensions   the sizes of the neighbor variables, corresponding to the order in neighborIndices
   * @param assignmentFeaturizer a function that maps from an assignment to the variables, represented as an array of
   *                             assignments in the same order as presented in neighborIndices, to a ConcatVector of
   *                             features for that assignment.
   *
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public VectorFactor addFactor(int[] neighborIndices, int[] neighborDimensions, Function<int[], ConcatVector> assignmentFeaturizer) {
    ConcatVectorTable features = new ConcatVectorTable(neighborDimensions);
    for (int[] assignment : features) {
      features.setAssignmentValue(assignment, () -> assignmentFeaturizer.apply(assignment));
    }

    return addFactor(features, neighborIndices);
  }

  /**
   * A simple helper function for defining a binary factor. That is, a factor between two variables in the graphical model.
   *
   * @param a The index of the first variable.
   * @param b The index of the second variable
   * @param featurizer The featurizer. This takes as input two assignments for the two variables, and returns the features on
  *                    those variables.
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public Factor addBinaryFactor(int a, int b, BiFunction<Integer, Integer, ConcatVector> featurizer) {
    int[] variableDims = getVariableSizes();
    assert a < variableDims.length;
    assert b < variableDims.length;
    return addFactor(new int[]{a, b}, new int[]{variableDims[a], variableDims[b]}, assignment -> featurizer.apply(assignment[0], assignment[1]) );
  }


  /**
   * Add a binary factor, with known dimensions for the variables
   * @param a The index of the first variable.
   * @param cardA The cardinality (i.e, dimension) of the first factor
   * @param b The index of the second variable
   * @param cardB The cardinality (i.e, dimension) of the second factor
   * @param featurizer The featurizer. This takes as input two assignments for the two variables, and returns the features on
   *                    those variables.
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public Factor addBinaryFactor(int a, int cardA, int b, int cardB, BiFunction<Integer, Integer, ConcatVector> featurizer) {
    return addFactor(new int[]{a, b}, new int[]{cardA, cardB}, assignment -> featurizer.apply(assignment[0], assignment[1]) );
  }


  /**
   * Add a binary factor, where we just want to hard-code the value of the factor.
   *
   * @param a The index of the first variable.
   * @param b The index of the second variable.
   * @param value A mapping from assignments of the two variables, to a factor value.
   *
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public Factor addStaticBinaryFactor(int a, int b, BiFunction<Integer, Integer, Double> value) {
    int[] variableDims = getVariableSizes();
    assert a < variableDims.length;
    assert b < variableDims.length;
    return addStaticFactor(new int[]{a, b}, new int[]{variableDims[a], variableDims[b]}, assignment ->
        value.apply(assignment[0], assignment[1]));
  }


  /**
   * Add a binary factor, where we just want to hard-code the value of the factor.
   *
   * @param a The index of the first variable.
   * @param cardA The cardinality (i.e, dimension) of the first factor
   * @param b The index of the second variable
   * @param cardB The cardinality (i.e, dimension) of the second factor
   * @param value A mapping from assignments of the two variables, to a factor value.
   *
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public Factor addStaticBinaryFactor(int a, int cardA, int b, int cardB, BiFunction<Integer, Integer, Double> value) {
    return addStaticFactor(new int[]{a, b}, new int[]{cardA, cardB}, assignment -> value.apply(assignment[0], assignment[1]));
  }


  /**
   * Set a training value for this variable in the graphical model.
   *
   * @param variable The variable to set.
   * @param value The value to set on the variable.
   */
  public void setTrainingLabel(int variable, int value) {
    getVariableMetaDataByReference(variable).put(LogLikelihoodDifferentiableFunction.VARIABLE_TRAINING_VALUE, Integer.toString(value));
  }

  /**
   * The next unused index which we can use for a new variable in a factor.
   */
  public int nextVariableIndex() {
    return numVariables();
  }


  /**
   * Creates an instantiated factor in this graph, with neighborIndices representing the neighbor variables by integer
   * index.
   *
   * @param featureTable    the feature table to use to drive the value of the factor
   * @param neighborIndices the indices of the neighboring variables, in order
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public VectorFactor addFactor(ConcatVectorTable featureTable, int[] neighborIndices) {
    assert (featureTable.getDimensions().length == neighborIndices.length);
    VectorFactor factor = new VectorFactor(featureTable, neighborIndices);
    factors.add(factor);
    return factor;
  }

  /**
   * Creates an instantiated factor in this graph, with neighborIndices representing the neighbor variables by integer
   * index.
   *
   * @param staticFeatureTable the feature table holding constant values for this factor
   * @param neighborIndices the indices of the neighboring variables, in order
   * @return a reference to the created factor. This can be safely ignored, as the factor is already saved in the model
   */
  public StaticFactor addStaticFactor(NDArrayDoubles staticFeatureTable, int[] neighborIndices) {
    assert (staticFeatureTable.getDimensions().length == neighborIndices.length);
    StaticFactor factor = new StaticFactor(staticFeatureTable, neighborIndices);
    factors.add(factor);
    return factor;
  }


  /**
   * Observe a given variable, setting it to a given value.
   *
   * @param variable The variable to set.
   * @param value The value we have observed this variable to have taken.
   */
  public void observe(int variable, int value) {
    getVariableMetaDataByReference(variable).put(CliqueTree.VARIABLE_OBSERVED_VALUE, Integer.toString(value));
  }


  /**
   * The number of variables in this graphical model.
   */
  public int numVariables() {
    int maxVar = 0;
    for (Factor f : factors) {
      for (int n : f.neigborIndices) {
        if (n > maxVar) maxVar = n;
      }
    }
    return maxVar + 1;
  }


  /**
   * @return an array of integers, indicating variable sizes given by each of the factors in the model
   */
  public int[] getVariableSizes() {
    if (factors.size() == 0) {
      return new int[0];
    }

    int maxVar = numVariables() - 1;
    int[] sizes = new int[maxVar + 1];
    for (int i = 0; i < sizes.length; i++) {
      sizes[i] = -1;
    }

    for (Factor f : factors) {
      for (int i = 0; i < f.neigborIndices.length; i++) {
        sizes[f.neigborIndices[i]] = f.getDimensions()[i];
      }
    }

    return sizes;
  }



  /**
   * Writes the protobuf version of this graphical model to a stream. reversible with readFromStream().
   *
   * @param stream the output stream to write to
   * @throws IOException passed through from the stream
   */
  public void writeToStream(OutputStream stream) throws IOException {
    getProtoBuilder().build().writeDelimitedTo(stream);
  }

  /**
   * Static function to deserialize a graphical model from an input stream.
   *
   * @param stream the stream to read from, assuming protobuf encoding
   * @return a new graphical model
   * @throws IOException passed through from the stream
   */
  public static GraphicalModel readFromStream(InputStream stream) throws IOException {
    return readFromProto(GraphicalModelProto.GraphicalModel.parseDelimitedFrom(stream));
  }

  /**
   * @return the proto builder corresponding to this GraphicalModel
   */
  public GraphicalModelProto.GraphicalModel.Builder getProtoBuilder() {
    GraphicalModelProto.GraphicalModel.Builder builder = GraphicalModelProto.GraphicalModel.newBuilder();
    builder.setMetaData(getProtoMetaDataBuilder(modelMetaData));
    for (Map<String, String> metaData : variableMetaData) {
      builder.addVariableMetaData(getProtoMetaDataBuilder(metaData));
    }
    for (Factor factor : factors) {
      builder.addFactor(factor.getProtoBuilder());
    }
    return builder;
  }

  /**
   * Recreates an in-memory GraphicalModel from a proto serialization, recursively creating all the ConcatVectorTable's
   * and ConcatVector's in memory as well.
   *
   * @param proto the proto to read
   * @return an in-memory GraphicalModel
   */
  public static GraphicalModel readFromProto(GraphicalModelProto.GraphicalModel proto) {
    if (proto == null) return null;
    GraphicalModel model = new GraphicalModel();
    model.modelMetaData = readMetaDataFromProto(proto.getMetaData());
    model.variableMetaData = new ArrayList<>();
    for (int i = 0; i < proto.getVariableMetaDataCount(); i++) {
      model.variableMetaData.add(readMetaDataFromProto(proto.getVariableMetaData(i)));
    }
    for (int i = 0; i < proto.getFactorCount(); i++) {
      model.factors.add(Factor.readFromProto(proto.getFactor(i)));
    }
    return model;
  }

  /**
   * Check that two models are deeply value-equivalent, down to the concat vectors inside the factor tables, within
   * some tolerance. Mostly useful for testing.
   *
   * @param other     the graphical model to compare against.
   * @param tolerance the tolerance to accept when comparing concat vectors for value equality.
   * @return whether the two models are tolerance equivalent
   */
  public boolean valueEquals(GraphicalModel other, double tolerance) {
    if (!modelMetaData.equals(other.modelMetaData)) {
      return false;
    }
    if (!variableMetaData.equals(other.variableMetaData)) {
      return false;
    }
    // compare factor sets for equality
    Set<Factor> remaining = new HashSet<>();
    remaining.addAll(factors);
    for (Factor otherFactor : other.factors) {
      Factor match = null;
      for (Factor factor : remaining) {
        if (factor.valueEquals(otherFactor, tolerance)) {
          match = factor;
          break;
        }
      }
      if (match == null) return false;
      else remaining.remove(match);
    }
    return remaining.size() <= 0;
  }

  /**
   * Displays a list of factors, by neighbor.
   *
   * @return a formatted list of factors, by neighbor
   */
  @Override
  public String toString() {
    String s = "{";
    for (Factor f : factors) {
      s += "\n\t" + Arrays.toString(f.neigborIndices) + "@" + f;
    }
    s += "\n}";
    return s;
  }

  /**
   * The point here is to allow us to save a copy of the model with a current set of factors and metadata mappings,
   * which can come in super handy with gameplaying applications. The cloned model doesn't instantiate the feature
   * thunks inside factors, those are just taken over individually.
   *
   * @return a clone
   */
  public GraphicalModel cloneModel() {
    GraphicalModel clone = new GraphicalModel();
    clone.modelMetaData.putAll(modelMetaData);
    for (int i = 0; i < variableMetaData.size(); i++) {
      if (variableMetaData.get(i) != null) {
        clone.getVariableMetaDataByReference(i).putAll(variableMetaData.get(i));
      }
    }

    for (Factor f : factors) {
      clone.factors.add(f.cloneFactor());
    }

    return clone;
  }

  ////////////////////////////////////////////////////////////////////////////
  // PRIVATE IMPLEMENTATION
  ////////////////////////////////////////////////////////////////////////////

  private static GraphicalModelProto.MetaData.Builder getProtoMetaDataBuilder(Map<String, String> metaData) {
    GraphicalModelProto.MetaData.Builder builder = GraphicalModelProto.MetaData.newBuilder();
    for (String key : metaData.keySet()) {
      builder.addKey(key);
      builder.addValue(metaData.get(key));
    }
    return builder;
  }

  private static Map<String, String> readMetaDataFromProto(GraphicalModelProto.MetaData proto) {
    Map<String, String> metaData = new HashMap<>();
    for (int i = 0; i < proto.getKeyCount(); i++) {
      metaData.put(proto.getKey(i), proto.getValue(i));
    }
    return metaData;
  }
}
