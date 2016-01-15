package com.github.keenon.loglinear.simple;

import com.github.keenon.loglinear.ConcatVectorProto;
import com.github.keenon.loglinear.GraphicalModelProto;
import com.github.keenon.loglinear.inference.CliqueTree;
import com.github.keenon.loglinear.learning.LogLikelihoodDifferentiableFunction;
import com.github.keenon.loglinear.model.ConcatVector;
import com.github.keenon.loglinear.model.ConcatVectorNamespace;
import com.github.keenon.loglinear.model.GraphicalModel;
import com.github.keenon.loglinear.simple.SimpleDurableModel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Created by keenon on 1/13/16.
 *
 * This is a high level API to simplify the common task of making a simple sequence model, with binary factors. There
 * are really three things going on here
 */
public class DurableSequenceModel extends SimpleDurableModel<Annotation> {
    public String[] tags;

    private Map<String, BiFunction<Annotation, Integer, String>> unaryStringFeatures = new HashMap<>();
    private Map<String, BiFunction<Annotation, Integer, double[]>> unaryEmbeddingFeatures = new HashMap<>();
    private Map<String, BiFunction<Annotation, Integer, String>> binaryStringFeatures = new HashMap<>();
    private StanfordCoreNLP coreNLP;

    private static final String SOURCE_TEXT = "com.github.keenon.loglinear.simple.DurableSequenceModel.SOURCE_TEXT";

    /**
     * This is the parent constructor that does the basic work of creating the backing model store, an optimizer, and
     * retraining synchronization infrastructure.
     *
     * @param backingStorePath the path to a folder where we can store backing information about the model
     * @param tags the tags we'll be using to classify the sequences into
     * @param coreNLP the instance of CoreNLP that we'll use to create any Annotation objects that we need
     */
    public DurableSequenceModel(String backingStorePath, String[] tags, StanfordCoreNLP coreNLP) throws IOException {
        super(backingStorePath);
        this.tags = tags;
        this.coreNLP = coreNLP;
    }

    /**
     * Grabs a set of labels off the loglinear classifier, without intervention
     * @param annotation the annotation to do labels for
     * @return an array with a single tag for each token in the annotation
     */
    public String[] labelSequence(Annotation annotation) {
        String[] tagSequence = new String[annotation.size()];

        GraphicalModel model = createModel(annotation);
        CliqueTree tree = new CliqueTree(model, weights);
        int[] map = tree.calculateMAP();

        for (int i = 0; i < tagSequence.length; i++) {
            tagSequence[i] = tags[map[i]];
        }

        return tagSequence;
    }

    /**
     * Create a training example, with the given labels, add it the classifier's set, and retrain (if we're not
     * retraining already). Also writes out to the durable log on disk.
     *
     * @param annotation the sentence
     * @param labels the gold labels for this sentence
     */
    public void addTrainingExample(Annotation annotation, String[] labels) {
        GraphicalModel model = createModel(annotation);
        for (int i = 0; i < labels.length; i++) {
            int tagIndex = -1;
            for (int t = 0; t <tags.length; t++) {
                if (tags[t].equals(labels[i])) {
                    tagIndex = t;
                }
            }
            assert(tagIndex != -1);
            model.getVariableMetaDataByReference(i).put(LogLikelihoodDifferentiableFunction.VARIABLE_TRAINING_VALUE, ""+tagIndex);
        }

        addLabeledTrainingExample(model);
    }

    /**
     * This adds a feature, which is a closure that takes an Annotation and an index into the sentence, and
     * returns a string value that's a feature on each unary factor in the sequence model.
     * @param name unique human readable name for the feature
     * @param newFeature the closure. must be idempotent
     */
    public void addUnaryStringFeature(String name, BiFunction<Annotation, Integer, String> newFeature) {
        unaryStringFeatures.put(name, newFeature);
    }

    /**
     * This adds a feature, which is a closure that takes an Annotation and an index into the sentence, and
     * returns a double array value (usually an embedding) that's a feature on each unary factor in the sequence model.
     * @param name unique human readable name for the feature
     * @param newFeature the closure. must be idempotent
     */
    public void addUnaryEmbeddingFeature(String name, BiFunction<Annotation, Integer, double[]> newFeature) {
        unaryEmbeddingFeatures.put(name, newFeature);
    }

    /**
     * This adds a feature, which is a closure that takes an Annotation and an index into the sentence, and
     * returns a string value that's a feature on each binary factor in the sequence model.
     * @param name unique human readable name for the feature
     * @param newFeature the closure. must be idempotent
     */
    public void addBinaryStringFeature(String name, BiFunction<Annotation, Integer, String> newFeature) {
        binaryStringFeatures.put(name, newFeature);
    }

    @Override
    protected GraphicalModel createModelInternal(Annotation annotation) {
        GraphicalModel model = new GraphicalModel();
        model.getModelMetaDataByReference().put(SOURCE_TEXT, annotation.toString());
        return model;
    }

    @Override
    protected Annotation restoreContextObjectFromModelTags(GraphicalModel model) {
        Annotation annotation = new Annotation(model.getModelMetaDataByReference().get(SOURCE_TEXT));
        coreNLP.annotate(annotation);
        return annotation;
    }

    @Override
    protected void featurizeModel(GraphicalModel model, Annotation annotation) {
        for (int i = 0; i < annotation.size(); i++) {
            final Integer f = i;

            // Add unary factor

            model.addFactor(new int[]{i}, new int[]{tags.length}, (assn) -> {
                ConcatVector unaryFeatureVector = namespace.newVector();
                String tag = tags[assn[0]];
                for (String feature : unaryStringFeatures.keySet()) {
                    String featureValue = unaryStringFeatures.get(feature).apply(annotation, f);
                    namespace.setSparseFeature(unaryFeatureVector, tag+":"+feature, featureValue, 1.0);
                }
                for (String feature : unaryEmbeddingFeatures.keySet()) {
                    double[] featureValue = unaryEmbeddingFeatures.get(feature).apply(annotation, f);
                    namespace.setDenseFeature(unaryFeatureVector, tag+":"+feature, featureValue);
                }
                return unaryFeatureVector;
            });

            // Add binary factor, if we're not at the end of the sequence

            if (i == annotation.size() - 1) continue;
            model.addFactor(new int[]{i, i+1}, new int[]{tags.length, tags.length}, (assn) -> {
                ConcatVector binaryFeatureVector = namespace.newVector();
                String leftTag = tags[assn[0]];
                String rightTag = tags[assn[1]];
                for (String feature : binaryStringFeatures.keySet()) {
                    String featureValue = binaryStringFeatures.get(feature).apply(annotation, f);
                    namespace.setSparseFeature(binaryFeatureVector, leftTag+":"+rightTag+":"+feature, featureValue, 1.0);
                }
                return binaryFeatureVector;
            });
        }
    }
}
