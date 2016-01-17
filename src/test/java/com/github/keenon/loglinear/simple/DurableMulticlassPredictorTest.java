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
public class DurableMulticlassPredictorTest {

    @Test
    public void testSequenceModel() throws Exception {
        String[] tags = new String[]{"happy", "sad"};

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        DurableMulticlassPredictor predictor = new DurableMulticlassPredictor("src/test/resources/multiclass", tags, pipeline);

        predictor.addStringFeature("first-token", ((annotation) -> annotation.get(CoreAnnotations.TokensAnnotation.class).get(0).word()));

        Annotation annotation = new Annotation("hello from Bob");
        pipeline.annotate(annotation);
        predictor.addTrainingExample(annotation, "happy");

        System.err.println(predictor.log.size());

        predictor.blockForRetraining();

        Annotation annotation2 = new Annotation("hello from Jane");
        pipeline.annotate(annotation2);
        System.err.println("-----\n\n");
        System.err.println(predictor.labelSequence(annotation2));
    }
}