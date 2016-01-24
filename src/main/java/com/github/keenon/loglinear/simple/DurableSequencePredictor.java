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
import java.util.function.BiFunction;

/**
 * Created by keenon on 1/13/16.
 *
 * This is a high level API to simplify the common task of making a simple sequence model, with binary factors.
 */
public class DurableSequencePredictor extends SimpleDurablePredictor<Annotation> {
    public String[] tags;

    private Map<String, BiFunction<Annotation, Integer, String>> unaryStringFeatures = new HashMap<>();
    private Map<String, BiFunction<Annotation, Integer, double[]>> unaryEmbeddingFeatures = new HashMap<>();
    private Map<String, BiFunction<Annotation, Integer, String>> binaryStringFeatures = new HashMap<>();
    private StanfordCoreNLP coreNLP;

    private static final String SOURCE_TEXT = "com.github.keenon.loglinear.simple.DurableSequencePredictor.SOURCE_TEXT";
    private static final String VARIABLE_TAG = "com.github.keenon.loglinear.simple.DurableSequencePredictor.VARIABLE_TAG";

    /**
     * This is the parent constructor that does the basic work of creating the backing model store, an optimizer, and
     * retraining synchronization infrastructure.
     *
     * @param backingStorePath the path to a folder where we can store backing information about the model
     * @param log the backing store to write saved data to / read data from
     * @param tags the tags we'll be using to classify the sequences into
     * @param coreNLP the instance of CoreNLP that we'll use to create any Annotation objects that we need
     */
    public DurableSequencePredictor(String backingStorePath, ModelLog log, String[] tags, StanfordCoreNLP coreNLP) throws IOException {
        super(backingStorePath, log);
        this.tags = tags;
        this.coreNLP = coreNLP;
    }

    /**
     * This is the parent constructor that does the basic work of creating the backing model store, an optimizer, and
     * retraining synchronization infrastructure.
     *
     * @param backingStorePath the path to a folder where we can store backing information about the model
     * @param tags the tags we'll be using to classify the sequences into
     * @param coreNLP the instance of CoreNLP that we'll use to create any Annotation objects that we need
     */
    public DurableSequencePredictor(String backingStorePath, String[] tags, StanfordCoreNLP coreNLP) throws IOException {
        super(backingStorePath, null);
        this.tags = tags;
        this.coreNLP = coreNLP;
        log = new SequenceModelLog(backingStorePath+"/data.tsv");
    }

    public class SequenceModelLog extends ModelLog {
        BufferedWriter bw;

        public SequenceModelLog(String path) throws IOException {
            File f = new File(path);
            if (!f.exists()) f.createNewFile();

            List<String> tokens = new ArrayList<>();
            List<String> labels = new ArrayList<>();

            BufferedReader br = new BufferedReader(new FileReader(path));
            String line;
            while (true) {
                line = br.readLine();
                String[] parts = new String[0];
                if (line != null) parts = line.split("\t");

                if (parts.length == 2) {
                    // Part of a continuing sentence, a token-label pair
                    tokens.add(parts[0]);
                    labels.add(parts[1]);
                }
                else {
                    if (tokens.size() > 0) {

                        // Create training example to add to the data set

                        StringBuilder sentenceBuilder = new StringBuilder();
                        for (int i = 0; i < tokens.size(); i++) {
                            if (i != 0) sentenceBuilder.append(" ");
                            sentenceBuilder.append(tokens.get(i));
                        }
                        Annotation annotation = new Annotation(sentenceBuilder.toString());
                        coreNLP.annotate(annotation);

                        add(createLabeledModel(annotation, labels.toArray(new String[labels.size()])), false);

                        // Clear the token sets to prepare for the next sentence

                        tokens.clear();
                        labels.clear();
                    }
                }

                if (line == null) break;
            }

            bw = new BufferedWriter(new FileWriter(path));
        }

        @Override
        public void writeExample(GraphicalModel m) throws IOException {
            Annotation annotation = context.get(m);
            for (int i = 0; i < annotation.get(CoreAnnotations.TokensAnnotation.class).size(); i++) {
                bw.write(annotation.get(CoreAnnotations.TokensAnnotation.class).get(i).word());
                bw.write("\t");
                bw.write(m.getVariableMetaDataByReference(i).get(VARIABLE_TAG));
                bw.write("\n");
            }
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
     * Grabs a set of labels off the loglinear classifier, without intervention
     * @param annotation the annotation to do labels for
     * @return an array with a single tag for each token in the annotation
     */
    public String[] labelSequence(Annotation annotation) {
        String[] tagSequence = new String[annotation.get(CoreAnnotations.TokensAnnotation.class).size()];

        GraphicalModel model = createModel(annotation);
        CliqueTree tree = new CliqueTree(model, weights);
        int[] map = tree.calculateMAP();

        for (int i = 0; i < tagSequence.length; i++) {
            tagSequence[i] = tags[map[i]];
        }

        return tagSequence;
    }

    /**
     * Builds a GraphicalModel for this Annotation object, and labels it.
     *
     * @param annotation the sentence
     * @param labels the gold labels for this sentence
     * @return a labeled GraphicalModel suitable for training models with
     */
    private GraphicalModel createLabeledModel(Annotation annotation, String[] labels) {
        GraphicalModel model = createModel(annotation);
        if (annotation.get(CoreAnnotations.TokensAnnotation.class).size() != labels.length) {
            throw new IllegalStateException("Shouldn't pass a training example with a"+
                    "different number of labels from tokens. Got a sentence \""+annotation+"\" with "+
                    annotation.get(CoreAnnotations.TokensAnnotation.class).size()+" tokens, but got labels "+
                    Arrays.toString(labels) + " with length "+labels.length);
        }
        assert(annotation.get(CoreAnnotations.TokensAnnotation.class).size() == labels.length);

        int maxVar = 0;
        for (GraphicalModel.Factor f : model.factors) {
            for (int i : f.neigborIndices) if (i > maxVar) maxVar = i;
        }
        if (maxVar+1 != labels.length) {
            System.err.println("Have the wrong number of labels!");
            throw new IllegalStateException();
        }

        for (int i = 0; i < labels.length; i++) {
            int tagIndex = -1;
            for (int t = 0; t < tags.length; t++) {
                if (tags[t].equals(labels[i])) {
                    tagIndex = t;
                }
            }
            assert(tagIndex != -1);
            model.getVariableMetaDataByReference(i).put(LogLikelihoodDifferentiableFunction.VARIABLE_TRAINING_VALUE, ""+tagIndex);
            model.getVariableMetaDataByReference(i).put(VARIABLE_TAG, labels[i]);
        }

        return model;
    }

    /**
     * Create a training example, with the given labels, add it the classifier's set, and retrain (if we're not
     * retraining already). Also writes out to the durable log on disk.
     *
     * @param annotation the sentence
     * @param labels the gold labels for this sentence
     */
    public void addTrainingExample(Annotation annotation, String[] labels) {
        addLabeledTrainingExample(createLabeledModel(annotation, labels));
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
        for (int i = 0; i < annotation.get(CoreAnnotations.TokensAnnotation.class).size(); i++) {
            final int f = i;

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

            if (i == annotation.get(CoreAnnotations.TokensAnnotation.class).size() - 1) continue;
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
