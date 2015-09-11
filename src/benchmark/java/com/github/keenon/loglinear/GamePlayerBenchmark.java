package com.github.keenon.loglinear;

import com.github.keenon.loglinear.model.ConcatVector;
import com.github.keenon.loglinear.model.GraphicalModel;

import java.io.IOException;
import java.util.*;

/**
 * Created by keenon on 9/11/15.
 *
 * This simulates game-player-like activity, with a few CoNLL CliqueTrees playing host to lots and lots of manipulations
 * by adding and removing human "observations". In real life, this kind of behavior occurs during sampling lookahead for
 * LENSE-like systems.
 *
 * In order to measure only the realistic parts of behavior, and not the random generation of numbers, we pre-cache a
 * few hundred ConcatVectors representing human obs features, then our feature function is just indexing into that cache.
 * The cache is designed to require a bit of L1 cache eviction to page through, so that we don't see artificial speed
 * gains during dot products b/c we already have both features and weights in L1 cache.
 */
public class GamePlayerBenchmark {
    public static void main(String[] args) throws IOException, ClassNotFoundException {

        //////////////////////////////////////////////////////////////
        // Generate the CoNLL CliqueTrees to use during gameplay
        //////////////////////////////////////////////////////////////

        CoNLLBenchmark coNLL = new CoNLLBenchmark();
        String prefix = System.getProperty("user.dir")+"/";
        if (prefix.endsWith("platform")) prefix = prefix+"learning/";

        List<CoNLLBenchmark.CoNLLSentence> train = coNLL.getSentences(prefix + "src/benchmark/data/conll.iob.4class.train");
        List<CoNLLBenchmark.CoNLLSentence> testA = coNLL.getSentences(prefix + "src/benchmark/data/conll.iob.4class.testa");
        List<CoNLLBenchmark.CoNLLSentence> testB = coNLL.getSentences(prefix + "src/benchmark/data/conll.iob.4class.testb");

        List<CoNLLBenchmark.CoNLLSentence> allData = new ArrayList<>();
        allData.addAll(train);
        allData.addAll(testA);
        allData.addAll(testB);

        Set<String> tagsSet = new HashSet<>();
        for (CoNLLBenchmark.CoNLLSentence sentence : allData) for (String nerTag : sentence.ner) tagsSet.add(nerTag);
        List<String> tags = new ArrayList<>();
        tags.addAll(tagsSet);

        coNLL.embeddings = coNLL.getEmbeddings(prefix + "src/benchmark/data/google-300-trimmed.ser.gz", allData);

        System.err.println("Making the training set...");

        int trainSize = train.size();
        GraphicalModel[] trainingSet = new GraphicalModel[trainSize];
        for (int i = 0; i < trainSize; i++) {
            if (i % 10 == 0) {
                System.err.println(i+"/"+trainSize);
            }
            trainingSet[i] = coNLL.generateSentenceModel(train.get(i), tags);
        }

        //////////////////////////////////////////////////////////////
        // Generate the random human observation feature vectors that we'll use
        //////////////////////////////////////////////////////////////

        Random r = new Random(10);
        int numFeatures = 5;
        int featureLength = 30;
        ConcatVector[] humanFeatureVectors = new ConcatVector[1000];
        for (int i = 0; i < humanFeatureVectors.length; i++) {
            humanFeatureVectors[i] = new ConcatVector(numFeatures);
            for (int j = 0; j < numFeatures; j++) {
                if (r.nextBoolean()) {
                    humanFeatureVectors[i].setSparseComponent(j, r.nextInt(featureLength), r.nextDouble());
                }
                else {
                    double[] dense = new double[featureLength];
                    for (int k = 0; k < dense.length; k++) {
                        dense[k] = r.nextDouble();
                    }
                    humanFeatureVectors[i].setDenseComponent(j, dense);
                }
            }
        }

        //////////////////////////////////////////////////////////////
        // Actually perform gameplay-like random mutations
        //////////////////////////////////////////////////////////////

        System.err.println("Warming up the JIT...");

        for (int i = 0; i < 100; i++) {
            System.err.println(i);
            gameplay(r, trainingSet[i], humanFeatureVectors);
        }

        System.err.println("Timing actual run...");

        long start = System.currentTimeMillis();
        for (int i = 0; i < 100; i++) {
            System.err.println(i);
            gameplay(r, trainingSet[i], humanFeatureVectors);
        }
        long duration = System.currentTimeMillis() - start;

        System.err.println("Duration: "+duration);
    }

    private static void gameplay(Random r, GraphicalModel model, ConcatVector[] humanFeatureVectors) {
        // TODO: implement something like TD-learning here
    }
}
