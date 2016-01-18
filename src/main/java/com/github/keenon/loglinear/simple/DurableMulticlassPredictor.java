package com.github.keenon.loglinear.simple;

import com.github.keenon.loglinear.inference.CliqueTree;
import com.github.keenon.loglinear.learning.LogLikelihoodDifferentiableFunction;
import com.github.keenon.loglinear.model.ConcatVector;
import com.github.keenon.loglinear.model.GraphicalModel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * Created by keenon on 1/17/16.
 *
 * This is a high level API to simplify the common task of making a simple multiclass model over text sequences.
 */
public class DurableMulticlassPredictor extends SimpleDurablePredictor<Annotation> {
    public String[] tags;

    private Map<String, Function<Annotation, String>> stringFeatures = new HashMap<>();
    private Map<String, Function<Annotation, double[]>> embeddingFeatures = new HashMap<>();
    private StanfordCoreNLP coreNLP;

    private static final String SOURCE_TEXT = "com.github.keenon.loglinear.simple.DurableMulticlassPredictor.SOURCE_TEXT";

    /**
     * This is the parent constructor that does the basic work of creating the backing model store, an optimizer, and
     * retraining synchronization infrastructure.
     *
     * @param backingStorePath the path to a folder where we can store backing information about the model
     */
    public DurableMulticlassPredictor(String backingStorePath, String[] tags, StanfordCoreNLP coreNLP) throws IOException {
        super(backingStorePath);

        this.tags = tags;
        this.coreNLP = coreNLP;
    }

    /**
     * This adds a feature, which is a closure that takes an Annotation and an index into the sentence, and
     * returns a string value that's a feature on each unary factor in the sequence model.
     * @param name unique human readable name for the feature
     * @param newFeature the closure. must be idempotent
     */
    public void addStringFeature(String name, Function<Annotation, String> newFeature) {
        stringFeatures.put(name, newFeature);
    }

    /**
     * This adds a feature, which is a closure that takes an Annotation and an index into the sentence, and
     * returns a double array value (usually an embedding) that's a feature on each unary factor in the sequence model.
     * @param name unique human readable name for the feature
     * @param newFeature the closure. must be idempotent
     */
    public void addEmbeddingFeature(String name, Function<Annotation, double[]> newFeature) {
        embeddingFeatures.put(name, newFeature);
    }

    /**
     * Grabs a set of labels off the loglinear classifier, without intervention
     * @param annotation the annotation to do labels for
     * @return the best tag for this sequence
     */
    public String labelSequence(Annotation annotation) {
        GraphicalModel model = createModel(annotation);
        CliqueTree tree = new CliqueTree(model, weights);
        int[] map = tree.calculateMAP();

        return tags[map[0]];
    }

    /**
     * Create a training example, with the given labels, add it the classifier's set, and retrain (if we're not
     * retraining already). Also writes out to the durable log on disk.
     *
     * @param annotation the sentence
     * @param label the gold label for this sentence
     */
    public void addTrainingExample(Annotation annotation, String label) {
        GraphicalModel model = createModel(annotation);

        int tagIndex = -1;
        for (int t = 0; t <tags.length; t++) {
            if (tags[t].equals(label)) {
                tagIndex = t;
            }
        }
        assert(tagIndex != -1);
        model.getVariableMetaDataByReference(0).put(LogLikelihoodDifferentiableFunction.VARIABLE_TRAINING_VALUE, ""+tagIndex);

        addLabeledTrainingExample(model);
    }

    @Override
    protected GraphicalModel createModelInternal(Annotation annotation) {
        GraphicalModel model = new GraphicalModel();
        model.getModelMetaDataByReference().put(SOURCE_TEXT, annotation.toString());
        featurizeModel(model, annotation);
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
        model.addFactor(new int[]{0}, new int[]{tags.length}, (assn) -> {
            ConcatVector featureVector = namespace.newVector();
            String tag = tags[assn[0]];
            for (String feature : stringFeatures.keySet()) {
                String featureValue = stringFeatures.get(feature).apply(annotation);
                namespace.setSparseFeature(featureVector, tag+":"+feature, featureValue, 1.0);
            }
            for (String feature : embeddingFeatures.keySet()) {
                double[] featureValue = embeddingFeatures.get(feature).apply(annotation);
                namespace.setDenseFeature(featureVector, tag+":"+feature, featureValue);
            }
            return featureVector;
        });
    }
}
