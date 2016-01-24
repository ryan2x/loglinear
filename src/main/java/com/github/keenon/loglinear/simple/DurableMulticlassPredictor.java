package com.github.keenon.loglinear.simple;

import com.github.keenon.loglinear.inference.CliqueTree;
import com.github.keenon.loglinear.learning.LogLikelihoodDifferentiableFunction;
import com.github.keenon.loglinear.model.ConcatVector;
import com.github.keenon.loglinear.model.GraphicalModel;
import com.github.keenon.loglinear.storage.ModelLog;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.*;
import java.util.*;
import java.util.function.Function;

/**
 * Created by keenon on 1/17/16.
 *
 * This is a high level API to simplify the common task of making a simple multiclass model over text sequences.
 */
public class DurableMulticlassPredictor extends SimpleDurablePredictor<Annotation> {
    public String[] tags;

    private Map<String, Function<Annotation, String>> stringFeatures = new HashMap<>();
    private Map<String, Function<Annotation, Set<String>>> setFeatures = new HashMap<>();
    private Map<String, Function<Annotation, double[]>> embeddingFeatures = new HashMap<>();
    private StanfordCoreNLP coreNLP;

    private static final String SOURCE_TEXT = "com.github.keenon.loglinear.simple.DurableMulticlassPredictor.SOURCE_TEXT";
    private static final String CLASS_LABEL = "com.github.keenon.loglinear.simple.DurableMulticlassPredictor.CLASS_LABEL";

    /**
     * This is the parent constructor that does the basic work of creating the backing model store, an optimizer, and
     * retraining synchronization infrastructure.
     *
     * @param backingStorePath the path to a folder where we can store backing information about the model
     */
    public DurableMulticlassPredictor(String backingStorePath, ModelLog log, String[] tags, StanfordCoreNLP coreNLP) throws IOException {
        super(backingStorePath, log);

        this.tags = tags;
        this.coreNLP = coreNLP;
    }

    /**
     * This is the parent constructor that does the basic work of creating the backing model store, an optimizer, and
     * retraining synchronization infrastructure.
     *
     * @param backingStorePath the path to a folder where we can store backing information about the model
     */
    public DurableMulticlassPredictor(String backingStorePath, String[] tags, StanfordCoreNLP coreNLP) throws IOException {
        super(backingStorePath, null);

        this.tags = tags;
        this.coreNLP = coreNLP;
        this.log = new MulticlassModelLog(backingStorePath+"/data.tsv");
    }

    public class MulticlassModelLog extends ModelLog {
        BufferedWriter bw;

        public MulticlassModelLog(String path) throws IOException {
            File f = new File(path);
            if (!f.exists()) f.createNewFile();

            BufferedReader br = new BufferedReader(new FileReader(path));
            String line;
            while (true) {
                line = br.readLine();
                String[] parts = new String[0];
                if (line != null) parts = line.split("\t");

                if (parts.length == 2) {
                    // Part of a continuing sentence, a token-label pair
                    String label = parts[0];
                    String sentence = parts[1];

                    Annotation annotation = new Annotation(sentence);
                    coreNLP.annotate(annotation);
                    add(createLabeledModel(annotation, label), false);
                }

                if (line == null) break;
            }

            bw = new BufferedWriter(new FileWriter(path, true)); // the true is for "append".
        }

        @Override
        public void writeExample(GraphicalModel m) throws IOException {
            bw.write(m.getModelMetaDataByReference().get(CLASS_LABEL));
            bw.write("\t");
            bw.write(m.getModelMetaDataByReference().get(SOURCE_TEXT).replaceAll("\t", "   "));
            bw.write("\n");
            bw.flush();
        }

        @Override
        public void close() throws IOException {
            bw.flush();
            bw.close();
        }
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
     * returns a set of string values that's a feature on each unary factor in the sequence model.
     * @param name unique human readable name for the feature
     * @param newFeature the closure. must be idempotent
     */
    public void addSetFeature(String name, Function<Annotation, Set<String>> newFeature) {
        setFeatures.put(name, newFeature);
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
     * Builds a GraphicalModel for this Annotation object, and a label to it
     *
     * @param annotation the sentence
     * @param label the gold label for this sentence
     * @return a labeled GraphicalModel suitable for training models with
     */
    private GraphicalModel createLabeledModel(Annotation annotation, String label) {
        GraphicalModel model = createModel(annotation);

        int tagIndex = -1;
        for (int t = 0; t <tags.length; t++) {
            if (tags[t].equals(label)) {
                tagIndex = t;
            }
        }
        assert(tagIndex != -1);
        model.getVariableMetaDataByReference(0).put(LogLikelihoodDifferentiableFunction.VARIABLE_TRAINING_VALUE, ""+tagIndex);
        model.getModelMetaDataByReference().put(CLASS_LABEL, label);

        return model;
    }

    /**
     * Create a training example, with the given labels, add it the classifier's set, and retrain (if we're not
     * retraining already). Also writes out to the durable log on disk.
     *
     * @param annotation the sentence
     * @param label the gold label for this sentence
     */
    public void addTrainingExample(Annotation annotation, String label) {
        addLabeledTrainingExample(createLabeledModel(annotation, label));
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
            for (String feature : setFeatures.keySet()) {
                Set<String> featureValue = setFeatures.get(feature).apply(annotation);
                for (String setElem : featureValue) {
                    namespace.setDenseFeature(featureVector, tag+":"+feature+":"+setElem, new double[]{1.0});
                }
            }
            for (String feature : embeddingFeatures.keySet()) {
                double[] featureValue = embeddingFeatures.get(feature).apply(annotation);
                namespace.setDenseFeature(featureVector, tag+":"+feature, featureValue);
            }
            return featureVector;
        });
    }
}
