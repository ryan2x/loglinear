package com.github.keenon.loglinear.model;

import static org.junit.Assert.*;

import com.carrotsearch.hppc.IntObjectHashMap;
import com.carrotsearch.hppc.IntObjectMap;
import com.carrotsearch.hppc.ObjectIntHashMap;
import com.carrotsearch.hppc.ObjectIntMap;
import com.carrotsearch.hppc.cursors.ObjectCursor;
import com.pholser.junit.quickcheck.ForAll;
import com.pholser.junit.quickcheck.From;
import com.pholser.junit.quickcheck.generator.GenerationStatus;
import com.pholser.junit.quickcheck.generator.Generator;
import com.pholser.junit.quickcheck.random.SourceOfRandomness;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.contrib.theories.Theories;
import org.junit.contrib.theories.Theory;
import org.junit.runner.RunWith;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;

/**
 * Created by keenon on 10/20/15.
 *
 * This checks the coherence of the ConcatVectorNamespace approach against the basic ConcatVector approach, using a cute
 * trick where we map random ints as "feature names", and double check that the output is always the same.
 */
@RunWith(Theories.class)
public class ConcatVectorNamespaceTest {

    @Ignore
    @Test
    public void testSpeed() {
        System.err.println("Initializing...");
        ConcatVectorNamespace namespace = new ConcatVectorNamespace();
        List<String> uuids = new ArrayList<>();
        for (int i = 0; i < 1000 * 1000; ++i) {
            String uuid = UUID.randomUUID().toString();
            namespace.ensureFeature(uuid);
            uuids.add(uuid);
        }
        System.err.println("Running...");
        long start = System.currentTimeMillis();
        long sum = 0;
        for (int i = 0; i < 1000 * 1000 * 1000; ++i) {
            sum += namespace.ensureFeature(uuids.get(i % uuids.size()));
        }
        System.err.println(sum);  // to prohibit JIT
        System.err.println("Took " + (System.currentTimeMillis() - start) / 1000 + " seconds");
    }

    @Theory
    public void testResizeOnSetComponent(@ForAll(sampleSize = 50) @From(MapGenerator.class) Map<Integer,Integer> featureMap1,
                                         @ForAll(sampleSize = 50) @From(MapGenerator.class) Map<Integer,Integer> featureMap2) {
        ConcatVectorNamespace namespace = new ConcatVectorNamespace();

        ConcatVector namespace1 = toNamespaceVector(namespace, featureMap1);
        ConcatVector namespace2 = toNamespaceVector(namespace, featureMap2);

        ConcatVector regular1 = toVector(featureMap1);
        ConcatVector regular2 = toVector(featureMap2);

        assertEquals(regular1.dotProduct(regular2), namespace1.dotProduct(namespace2), 1.0e-5);

        ConcatVector namespaceSum = namespace1.deepClone();
        namespaceSum.addVectorInPlace(namespace2, 1.0);

        ConcatVector regularSum = regular1.deepClone();
        regularSum.addVectorInPlace(regular2, 1.0);

        assertEquals(regular1.dotProduct(regularSum), namespace1.dotProduct(namespaceSum), 1.0e-5);
        assertEquals(regularSum.dotProduct(regular2), namespaceSum.dotProduct(namespace2), 1.0e-5);
    }

    @Theory
    public void testProtoNamespace(@ForAll(sampleSize = 50) @From(NamespaceGenerator.class) ConcatVectorNamespace namespace) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        namespace.writeToStream(byteArrayOutputStream);
        byteArrayOutputStream.close();

        byte[] bytes = byteArrayOutputStream.toByteArray();

        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);

        ConcatVectorNamespace recovered = ConcatVectorNamespace.readFromStream(byteArrayInputStream);

        assertEquals(namespace.featureToIndex, recovered.featureToIndex);
        assertEquals(namespace.sparseFeatureIndex, recovered.sparseFeatureIndex);
        assertEquals(namespace.reverseSparseFeatureIndex, recovered.reverseSparseFeatureIndex);
    }

    public ConcatVector toNamespaceVector(ConcatVectorNamespace namespace, Map<Integer,Integer> featureMap) {
        ConcatVector newVector = namespace.newVector();
        for (int i : featureMap.keySet()) {
            String feature = "feat"+i;
            String sparse = "index"+featureMap.get(i);
            namespace.setSparseFeature(newVector, feature, sparse, 1.0);
        }
        return newVector;
    }

    public ConcatVector toVector(Map<Integer, Integer> featureMap) {
        ConcatVector vector = new ConcatVector(20);
        for (int i : featureMap.keySet()) {
            vector.setSparseComponent(i, featureMap.get(i), 1.0);
        }
        return vector;
    }

    public static class MapGenerator extends Generator<Map<Integer,Integer>> {
        public MapGenerator(Class<Map<Integer, Integer>> type) {
            super(type);
        }

        @Override
        public Map<Integer, Integer> generate(SourceOfRandomness sourceOfRandomness, GenerationStatus generationStatus) {
            int numFeatures = sourceOfRandomness.nextInt(1,15);
            Map<Integer, Integer> featureMap = new HashMap<>();
            for (int i = 0; i < numFeatures; i++) {
                int featureValue = sourceOfRandomness.nextInt(20);
                while (featureMap.containsKey(featureValue)) {
                    featureValue = sourceOfRandomness.nextInt(20);
                }

                featureMap.put(featureValue, sourceOfRandomness.nextInt(2));
            }
            return featureMap;
        }
    }

    public static class NamespaceGenerator extends Generator<ConcatVectorNamespace> {
        public NamespaceGenerator(Class<ConcatVectorNamespace> type) {
            super(type);
        }

        private ObjectIntMap<String> generateFeatureMap(SourceOfRandomness sourceOfRandomness) {
            ObjectIntMap<String> featureMap = new ObjectIntHashMap<>();
            int numFeatures = sourceOfRandomness.nextInt(1,15);
            for (int i = 0; i < numFeatures; i++) {
                int featureValue = sourceOfRandomness.nextInt(20);
                while (featureMap.containsKey(""+featureValue)) {
                    featureValue = sourceOfRandomness.nextInt(20);
                }

                featureMap.put(""+featureValue, sourceOfRandomness.nextInt(2));
            }
            return featureMap;
        }

        @Override
        public ConcatVectorNamespace generate(SourceOfRandomness sourceOfRandomness, GenerationStatus generationStatus) {
            ConcatVectorNamespace namespace = new ConcatVectorNamespace();
            namespace.featureToIndex.putAll(generateFeatureMap(sourceOfRandomness));
            for (ObjectCursor<String> key : namespace.featureToIndex.keys()) {
                if (sourceOfRandomness.nextBoolean()) {
                    ObjectIntMap<String> sparseMap = generateFeatureMap(sourceOfRandomness);
                    IntObjectMap< String> reverseSparseMap = new IntObjectHashMap<>();
                    for (ObjectCursor<String> sparseKey : sparseMap.keys()) reverseSparseMap.put(sparseMap.get(sparseKey.value), sparseKey.value);
                    namespace.sparseFeatureIndex.put(key.value, sparseMap);
                    namespace.reverseSparseFeatureIndex.put(key.value, reverseSparseMap);
                }
            }
            return namespace;
        }
    }
}