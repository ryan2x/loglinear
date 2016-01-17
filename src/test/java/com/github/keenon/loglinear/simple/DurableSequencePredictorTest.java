package com.github.keenon.loglinear.simple;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.junit.Test;

import java.util.Arrays;
import java.util.Properties;

/**
 * Created by keenon on 1/15/16.
 *
 * Tests for DurableSequencePredictor, which may not actually be very complex or randomized.
 */
public class DurableSequencePredictorTest {

    @Test
    public void testSequenceModel() throws Exception {
        String[] tags = new String[]{"person", "none"};

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        DurableSequencePredictor predictor = new DurableSequencePredictor("src/test/resources/sequence", tags, pipeline);

        predictor.addUnaryStringFeature("token", ((annotation, index) -> annotation.get(CoreAnnotations.TokensAnnotation.class).get(index).word()));

        Annotation annotation = new Annotation("hello from Bob");
        pipeline.annotate(annotation);
        String[] labels = new String[]{"none", "none", "person"};
        predictor.addTrainingExample(annotation, labels);

        System.err.println(predictor.log.size());

        predictor.blockForRetraining();

        Annotation annotation2 = new Annotation("Bob from hello");
        pipeline.annotate(annotation2);
        System.err.println("-----\n\n");
        System.err.println(Arrays.toString(predictor.labelSequence(annotation2)));
    }
}