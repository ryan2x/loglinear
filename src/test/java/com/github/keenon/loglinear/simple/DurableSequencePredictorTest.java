package com.github.keenon.loglinear.simple;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Random;

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

        predictor.addBinaryStringFeature("bias", (((annotation, integer) -> "true")));
        predictor.addUnaryStringFeature("token", ((annotation, index) -> {
            if (index >= annotation.get(CoreAnnotations.TokensAnnotation.class).size()) {
                System.err.println("Request for "+index+" in sentence "+annotation);
                throw new IllegalStateException();
            }
            return annotation.get(CoreAnnotations.TokensAnnotation.class).get(index).word();
        }));

        Annotation annotation = new Annotation("hello from Bob");
        pipeline.annotate(annotation);
        String[] labels = new String[]{"none", "none", "person"};
        predictor.addTrainingExample(annotation, labels);

        System.err.println(predictor.log.size());

        predictor.blockForRetraining();

        Random r = new Random(400);
        String[] tokenSource = new String[]{
                "Bob",
                "Jill",
                "Greg",
                "hello",
                "from",
                "and"
        };
        String[] labelSource = new String[]{
                "person",
                "person",
                "person",
                "none",
                "none",
                "none"
        };
        for (int i = 0; i < 200; i++) {
            int len = r.nextInt(8)+2;
            String tokens = "";
            String[] labels2 = new String[len];
            for (int j = 0; j < len; j++) {
                int l = r.nextInt(tokenSource.length);
                if (j > 0) tokens += " ";
                tokens += tokenSource[l];
                labels2[j] = labelSource[l];
            }
            Annotation annotation2 = new Annotation(tokens);
            pipeline.annotate(annotation2);
            predictor.addTrainingExample(annotation2, labels2);
        }

        predictor.blockForRetraining();

        Annotation annotation3 = new Annotation("Bob from hello");
        pipeline.annotate(annotation3);
        System.err.println("-----\n\n");
        System.err.println(Arrays.toString(predictor.labelSequence(annotation3)));
    }
}