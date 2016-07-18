package com.github.keenon.loglinear.model;

import com.pholser.junit.quickcheck.ForAll;
import com.pholser.junit.quickcheck.From;
import com.pholser.junit.quickcheck.generator.GenerationStatus;
import com.pholser.junit.quickcheck.generator.Generator;
import com.pholser.junit.quickcheck.random.SourceOfRandomness;
import org.junit.contrib.theories.Theories;
import org.junit.contrib.theories.Theory;
import org.junit.runner.RunWith;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertTrue;

/**
 * Created by keenon on 7/12/16.
 *
 * This tests functionality in NDArrayDoubles that's special, which is basically just the ability to write to proto.
 */
@RunWith(Theories.class)
public class NDArrayDoublesTest {

  @Theory
  public void testProto(@ForAll(sampleSize = 50) @From(NDArrayGenerator.class) NDDoubleArrayWithGold testPair) throws Exception {
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

    testPair.array.writeToStream(byteArrayOutputStream);
    byteArrayOutputStream.close();

    byte[] bytes = byteArrayOutputStream.toByteArray();

    ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);

    NDArrayDoubles recovered = NDArrayDoubles.readFromStream(byteArrayInputStream);

    assertTrue(testPair.array.valueEquals(recovered, 1.0e-5));
  }

  public static class NDDoubleArrayWithGold {
    public NDArrayDoubles array;
    public Map<int[], Double> gold = new HashMap<>();
  }

  public static class NDArrayGenerator extends Generator<NDDoubleArrayWithGold> {
    public NDArrayGenerator(Class<NDDoubleArrayWithGold> type) {
      super(type);
    }

    @Override
    public NDDoubleArrayWithGold generate(SourceOfRandomness sourceOfRandomness, GenerationStatus generationStatus) {
      NDDoubleArrayWithGold testPair = new NDDoubleArrayWithGold();

      int numDimensions = sourceOfRandomness.nextInt(1, 5);
      int[] dimensions = new int[numDimensions];
      for (int i = 0; i < dimensions.length; i++) {
        dimensions[i] = sourceOfRandomness.nextInt(1, 4);
      }

      testPair.array = new NDArrayDoubles(dimensions);
      recursivelyFillArray(new ArrayList<>(), testPair, sourceOfRandomness);

      return testPair;
    }

    private static void recursivelyFillArray(List<Integer> assignmentSoFar, NDDoubleArrayWithGold testPair, SourceOfRandomness sourceOfRandomness) {
      if (assignmentSoFar.size() == testPair.array.getDimensions().length) {
        int[] arr = new int[assignmentSoFar.size()];
        for (int i = 0; i < arr.length; i++) {
          arr[i] = assignmentSoFar.get(i);
        }

        double value = sourceOfRandomness.nextDouble();
        testPair.array.setAssignmentValue(arr, value);
        testPair.gold.put(arr, value);
      } else {
        for (int i = 0; i < testPair.array.getDimensions()[assignmentSoFar.size()]; i++) {
          List<Integer> newList = new ArrayList<>();
          newList.addAll(assignmentSoFar);
          newList.add(i);
          recursivelyFillArray(newList, testPair, sourceOfRandomness);
        }
      }
    }
  }
}