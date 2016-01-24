package com.github.keenon.loglinear.storage;

import com.github.keenon.loglinear.model.GraphicalModel;
import com.github.keenon.loglinear.model.GraphicalModelTest;
import com.pholser.junit.quickcheck.ForAll;
import com.pholser.junit.quickcheck.From;
import com.pholser.junit.quickcheck.generator.GenerationStatus;
import com.pholser.junit.quickcheck.generator.Generator;
import com.pholser.junit.quickcheck.random.SourceOfRandomness;
import org.junit.contrib.theories.Theories;
import org.junit.contrib.theories.Theory;
import org.junit.runner.RunWith;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Created by keenon on 10/17/15.
 *
 * This just double checks that we can write and read these model batches without loss.
 */
@RunWith(Theories.class)
public class ModelLogDiskTest {
    @Theory
    public void testProtoBatch(@ForAll(sampleSize = 50) @From(ModelBatchTest.BatchGenerator.class) ModelBatch batch) throws IOException {
        ByteArrayInputStream emptyInputStream = new ByteArrayInputStream(new byte[0]);
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        ModelLog log = new ModelLogDisk(emptyInputStream, byteArrayOutputStream);
        log.writeWithFactors = true;
        for (GraphicalModel model : batch) {
            log.add(model);
        }
        log.close();

        byte[] bytes = byteArrayOutputStream.toByteArray();
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);

        ModelLog recovered = new ModelLogDisk(byteArrayInputStream, byteArrayOutputStream);
        byteArrayInputStream.close();

        assertEquals(batch.size(), recovered.size());

        for (int i = 0; i < batch.size(); i++) {
            assertTrue(batch.get(i).valueEquals(recovered.get(i), 1.0e-5));
        }
    }

    @Theory
    public void testProtoBatchCloseContinue(@ForAll(sampleSize = 50) @From(ModelBatchTest.BatchGenerator.class) ModelBatch batch) throws IOException {
        ByteArrayInputStream emptyInputStream = new ByteArrayInputStream(new byte[0]);
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        ModelLog log = new ModelLogDisk(emptyInputStream, byteArrayOutputStream);
        log.writeWithFactors = true;
        for (GraphicalModel model : batch) {
            log.add(model);
        }
        log.close();

        byte[] bytes = byteArrayOutputStream.toByteArray();
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);

        ByteArrayOutputStream secondByteArrayOutputStream = new ByteArrayOutputStream();
        secondByteArrayOutputStream.write(bytes);
        ModelLog secondLog = new ModelLogDisk(byteArrayInputStream, secondByteArrayOutputStream);
        secondByteArrayOutputStream.close();

        secondLog.writeWithFactors = true;
        for (GraphicalModel model : batch) {
            secondLog.add(model);
        }
        secondLog.close();
        assertEquals(batch.size() * 2, secondLog.size());

        byte[] finalBytes = secondByteArrayOutputStream.toByteArray();
        ByteArrayInputStream secondByteArrayInputStream = new ByteArrayInputStream(finalBytes);

        ModelLog recovered = new ModelLogDisk(secondByteArrayInputStream, byteArrayOutputStream);
        secondByteArrayInputStream.close();

        assertEquals(batch.size()*2, recovered.size());

        for (int i = 0; i < batch.size(); i++) {
            assertTrue(batch.get(i).valueEquals(recovered.get(i), 1.0e-5));
        }
        for (int i = batch.size(); i < batch.size()*2; i++) {
            assertTrue(batch.get(i-batch.size()).valueEquals(recovered.get(i), 1.0e-5));
        }
    }

}