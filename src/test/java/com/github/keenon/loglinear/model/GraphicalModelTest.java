package com.github.keenon.loglinear.model;

import com.github.keenon.loglinear.inference.CliqueTree;
import com.pholser.junit.quickcheck.ForAll;
import com.pholser.junit.quickcheck.From;
import com.pholser.junit.quickcheck.generator.GenerationStatus;
import com.pholser.junit.quickcheck.generator.Generator;
import com.pholser.junit.quickcheck.random.SourceOfRandomness;
import org.junit.Test;
import org.junit.contrib.theories.Theories;
import org.junit.contrib.theories.Theory;
import org.junit.runner.RunWith;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by keenon on 8/11/15.
 * <p>
 * Quickchecks a couple of pieces of functionality, but mostly the serialization and deserialization (basically the only
 * non-trivial section).
 */
@RunWith(Theories.class)
public class GraphicalModelTest {


  @Theory
  public void testProtoModel(@ForAll(sampleSize = 50) @From(GraphicalModelGenerator.class) GraphicalModel graphicalModel) throws IOException {
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

    graphicalModel.writeToStream(byteArrayOutputStream);
    byteArrayOutputStream.close();

    byte[] bytes = byteArrayOutputStream.toByteArray();

    ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);

    GraphicalModel recovered = GraphicalModel.readFromStream(byteArrayInputStream);

    assertTrue(graphicalModel.valueEquals(recovered, 1.0e-5));
  }


  @Test
  public void testAddNormalizedFactorSingleton() {
    GraphicalModel model = new GraphicalModel();
    model.addStaticNormalizedFactor(new int[]{0}, new int[]{3}, assgn -> 42.0);
    assertEquals(1.0, new CliqueTree(model, new ConcatVector(0)).calculateMarginals().partitionFunction, 1e-9);

    Random r = new Random(42L);
    for (int i = 0; i < 1000; ++i) {
      GraphicalModel fuzz = new GraphicalModel();
      double[] assign = new double[3];
      for (int k = 0; k < assign.length; ++k) {
        assign[k] = r.nextDouble();
      }
      fuzz.addStaticNormalizedFactor(new int[]{0}, new int[]{3}, k -> assign[k[0]]);
      assertEquals(1.0, new CliqueTree(fuzz, new ConcatVector(0)).calculateMarginals().partitionFunction, 1e-9);
    }
  }


  @Test
  public void testAddNormalizedFactorSingletonLargeMagnitude() {
    GraphicalModel model = new GraphicalModel();
    model.addStaticNormalizedFactor(new int[]{0}, new int[]{3}, assgn -> 42.0);
    assertEquals(1.0, new CliqueTree(model, new ConcatVector(0)).calculateMarginals().partitionFunction, 1e-9);

    Random r = new Random(42L);
    for (int i = 0; i < 1000; ++i) {
      GraphicalModel fuzz = new GraphicalModel();
      double[] assign = new double[3];
      for (int k = 0; k < assign.length; ++k) {
        assign[k] = r.nextDouble();
      }
      fuzz.addStaticNormalizedFactor(new int[]{0}, new int[]{3}, k -> assign[k[0]] * 10000.0);
      assertEquals(1.0, new CliqueTree(fuzz, new ConcatVector(0)).calculateMarginals().partitionFunction, 1e-9);
    }
  }


  @Test
  public void testAddNormalizedFactorBinary() {
    Random r = new Random(42L);
    for (int i = 0; i < 1000; ++i) {
      GraphicalModel fuzz = new GraphicalModel();
      double[][] assign = new double[3][3];
      for (int k = 0; k < assign.length; ++k) {
        for (int l = 0; l < assign.length; ++l) {
          assign[k][l] = r.nextDouble();
        }
      }
      fuzz.addStaticNormalizedFactor(new int[]{0, 1}, new int[]{3,3}, arr -> assign[arr[0]][arr[1]]);
      assertEquals(1.0, new CliqueTree(fuzz, new ConcatVector(0)).calculateMarginals().partitionFunction, 1e-9);
    }
  }


  /**
   * Create a single conditional factor, and make sure it normalizes correctly
   */
  @Test
  public void testAddStaticConditionalFactor() {
    Random r = new Random(42L);
    for (int i = 0; i < 1000; ++i) {
      GraphicalModel fuzz = new GraphicalModel();
      double[][] assign = new double[3][3];
      for (int k = 0; k < assign.length; ++k) {
        for (int l = 0; l < assign.length; ++l) {
          assign[k][l] = r.nextDouble();
        }
      }
      fuzz.addStaticConditionalFactor(new int[]{0}, 1, new int[]{3}, 3, (arr, cons)  -> assign[arr[0]][cons]);

      // 1. Test the partition function is |consequent|
      assertEquals(3.0, new CliqueTree(fuzz, new ConcatVector(0)).calculateMarginals().partitionFunction, 1e-9);

      // 2. Test the partition function is 1 if the antecedent is observed
      fuzz.observe(0, 0);
      assertEquals(1.0, new CliqueTree(fuzz, new ConcatVector(0)).calculateMarginals().partitionFunction, 1e-9);
    }
  }


  /**
   * Create a simple chain Bayes net, without a prior on the root variable.
   */
  @Test
  public void testCreateBayesNet() {
    Random r = new Random(42L);
    for (int i = 0; i < 1000; ++i) {
      GraphicalModel fuzz = new GraphicalModel();

      // Add the first factor
      double[][] factorA = new double[3][3];
      for (int k = 0; k < factorA.length; ++k) {
        for (int l = 0; l < factorA.length; ++l) {
          factorA[k][l] = r.nextDouble();
        }
      }
      fuzz.addStaticConditionalFactor(new int[]{0}, 1, new int[]{3}, 3, (arr, cons)  -> factorA[arr[0]][cons]);

      // Add the second factor
      double[][] factorB = new double[3][5];
      for (int k = 0; k < factorB.length; ++k) {
        for (int l = 0; l < factorB.length; ++l) {
          factorB[k][l] = r.nextDouble();
        }
      }
      fuzz.addStaticConditionalFactor(new int[]{1}, 2, new int[]{3}, 5, (arr, cons)  -> factorB[arr[0]][cons]);

      fuzz.observe(0, 0);
      assertEquals(1.0, new CliqueTree(fuzz, new ConcatVector(0)).calculateMarginals().partitionFunction, 1e-9);
    }
  }



  @Theory
  public void testClone(@ForAll(sampleSize = 50) @From(GraphicalModelGenerator.class) GraphicalModel graphicalModel) throws IOException {
    GraphicalModel clone = graphicalModel.cloneModel();
    assertTrue(graphicalModel.valueEquals(clone, 1.0e-5));
  }


  @Theory
  public void testGetVariableSizes(@ForAll(sampleSize = 50) @From(GraphicalModelGenerator.class) GraphicalModel graphicalModel) throws IOException {
    int[] sizes = graphicalModel.getVariableSizes();

    for (GraphicalModel.Factor f : graphicalModel.factors) {
      for (int i = 0; i < f.neigborIndices.length; i++) {
        assertEquals(f.getDimensions()[i], sizes[f.neigborIndices[i]]);
      }
    }
  }


  public static class GraphicalModelGenerator extends Generator<GraphicalModel> {
    public GraphicalModelGenerator(Class<GraphicalModel> type) {
      super(type);
    }

    private Map<String, String> generateMetaData(SourceOfRandomness sourceOfRandomness, Map<String, String> metaData) {
      int numPairs = sourceOfRandomness.nextInt(9);
      for (int i = 0; i < numPairs; i++) {
        int key = sourceOfRandomness.nextInt();
        int value = sourceOfRandomness.nextInt();
        metaData.put("key:" + key, "value:" + value);
      }
      return metaData;
    }

    @Override
    public GraphicalModel generate(SourceOfRandomness sourceOfRandomness, GenerationStatus generationStatus) {
      GraphicalModel model = new GraphicalModel();

      // Create the variables and factors

      int[] variableSizes = new int[20];
      for (int i = 0; i < 20; i++) {
        variableSizes[i] = sourceOfRandomness.nextInt(1, 5);
      }
      int numFactors = sourceOfRandomness.nextInt(12);
      for (int i = 0; i < numFactors; i++) {
        int[] neighbors = new int[sourceOfRandomness.nextInt(1, 3)];
        int[] neighborSizes = new int[neighbors.length];
        for (int j = 0; j < neighbors.length; j++) {
          neighbors[j] = sourceOfRandomness.nextInt(20);
          neighborSizes[j] = variableSizes[neighbors[j]];
        }

        ConcatVectorTable table = new ConcatVectorTable(neighborSizes);
        for (int[] assignment : table) {
          int numComponents = sourceOfRandomness.nextInt(7);
          // Generate a vector
          ConcatVector v = new ConcatVector(numComponents);
          for (int x = 0; x < numComponents; x++) {
            if (sourceOfRandomness.nextBoolean()) {
              v.setSparseComponent(x, sourceOfRandomness.nextInt(32), sourceOfRandomness.nextDouble());
            } else {
              double[] val = new double[sourceOfRandomness.nextInt(12)];
              for (int y = 0; y < val.length; y++) {
                val[y] = sourceOfRandomness.nextDouble();
              }
              v.setDenseComponent(x, val);
            }
          }
          // set vec in table
          table.setAssignmentValue(assignment, () -> v);
        }

        model.addFactor(table, neighbors);
      }

      // Add metadata to the variables, factors, and model

      generateMetaData(sourceOfRandomness, model.getModelMetaDataByReference());
      for (int i = 0; i < 20; i++) {
        generateMetaData(sourceOfRandomness, model.getVariableMetaDataByReference(i));
      }
      for (GraphicalModel.Factor factor : model.factors) {
        generateMetaData(sourceOfRandomness, factor.getMetaDataByReference());
      }

      return model;
    }
  }
}