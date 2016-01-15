package com.github.keenon.loglinear.simple;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.junit.Test;

import java.util.Arrays;
import java.util.Properties;

import static org.junit.Assert.*;

/**
 * Created by keenon on 1/15/16.
 *
 * Tests for DurableSequenceModel, which may not actually be very complex or randomized.
 */
public class DurableSequenceModelTest {

    @Test
    public void testSequenceModel() throws Exception {
        String[] tags = new String[]{"person", "none"};

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        DurableSequenceModel model = new DurableSequenceModel("src/test/resources/sequence", tags, pipeline);

        model.addUnaryStringFeature("token", ((annotation, index) -> annotation.get(CoreAnnotations.TokensAnnotation.class).get(index).word()));

        Annotation annotation = new Annotation("hello from Bob");
        pipeline.annotate(annotation);
        String[] labels = new String[]{"none", "none", "person"};
        model.addTrainingExample(annotation, labels);

        System.err.println(model.log.size());

        Thread.sleep(20000);

        Annotation annotation2 = new Annotation("hello from Bob");
        pipeline.annotate(annotation2);
        System.err.println(Arrays.toString(model.labelSequence(annotation2)));
    }
}