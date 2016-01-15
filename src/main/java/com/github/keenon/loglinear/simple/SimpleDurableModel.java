package com.github.keenon.loglinear.simple;

import com.github.keenon.loglinear.learning.AbstractBatchOptimizer;
import com.github.keenon.loglinear.learning.BacktrackingAdaGradOptimizer;
import com.github.keenon.loglinear.learning.LogLikelihoodDifferentiableFunction;
import com.github.keenon.loglinear.model.ConcatVector;
import com.github.keenon.loglinear.model.ConcatVectorNamespace;
import com.github.keenon.loglinear.model.GraphicalModel;
import com.github.keenon.loglinear.storage.ModelLog;

import java.io.*;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by keenon on 1/13/16.
 *
 * We have this interface so that users of these SimpleDurableModel's (like LENSE) can have a unified interface to talk to
 * several different kinds of SimpleDurableModel.
 *
 * We have multiple SimpleDurableModel types as a simple interface for downstream users who don't want the full
 * complexity available in the real interface.
 */
public abstract class SimpleDurableModel<T extends Serializable> {
    public ConcatVector weights;

    protected ConcatVectorNamespace namespace;
    protected ModelLog log;
    protected Map<GraphicalModel, T> context = new IdentityHashMap<>();

    private String weightsPath;
    private String namespacePath;
    private boolean trainingRunning = false;

    /**
     * This is the parent constructor that does the basic work of creating the backing model store, an optimizer, and
     * retraining synchronization infrastructure.
     *
     * @param backingStorePath the path to a folder where we can store backing information about the model
     */
    public SimpleDurableModel(String backingStorePath) throws IOException {
        log = new ModelLog(backingStorePath+"/model-log.ser");

        // Check if the weights have a complete, valid serialized form on the backing store, and load

        weightsPath = backingStorePath+"/weights.ser";
        try {
            File f = new File(weightsPath);
            if (f.exists()) {
                InputStream is = new FileInputStream(weightsPath);
                weights = ConcatVector.readFromStream(is);
            }
            else {
                // This means that the weights didn't exist yet
                weights = new ConcatVector(1);
            }
        }
        catch (Exception e) {
            System.err.println("weights.ser is corrupted, unable to read it");
            weights = new ConcatVector(1);
        }

        // Check if the namespace has a complete, valid serialized form on the backing store, and load

        namespacePath = backingStorePath+"/namespace.ser";
        try {
            File f = new File(namespacePath);
            if (f.exists()) {
                ObjectInputStream is = new ObjectInputStream(new FileInputStream(namespacePath));
                namespace = (ConcatVectorNamespace)is.readObject();
            }
            else {
                // This means that the namespace doesn't exist yet
                namespace = new ConcatVectorNamespace();
            }
        }
        catch (Exception e) {
            System.err.println("namespace.ser is corrupted, unable to read it");
            namespace = new ConcatVectorNamespace();
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // Interfaces for integrating packages like LENSE
    ////////////////////////////////////////////////////////////////////////

    /**
     * IF YOU ARE AN END-USER, THIS ISN'T WHAT YOU'RE LOOKING FOR
     *
     * This needs to get implemented by subclasses, and is the primary complexity hiding mechanism for external apps
     * that intend to integrate with the SimpleDurableModel interface, like Lense.
     *
     * @param t the input type that gives us the information to featurize a model
     * @return a GraphicalModel that's fully featurized
     */
    public GraphicalModel createModel(T t) {
        GraphicalModel model = createModelInternal(t);
        context.put(model, t);
        return model;
    }

    protected abstract GraphicalModel createModelInternal(T t);

    ////////////////////////////////////////////////////////////////////////
    // Protected interfaces only for subclasses
    ////////////////////////////////////////////////////////////////////////

    /**
     * Puts an additional training example into the ModelLog, which writes to disk, and then kicks off a retraining
     * job if there aren't any currently running.
     *
     * @param model the model to add to the training set
     */
    protected void addLabeledTrainingExample(GraphicalModel model) {
        log.add(model);
        launchTrainingRunIfNotRunning();
    }

    /**
     * This is for subclasses, and is used to re-create any in-memory contextual information that might get stored
     * along with the GraphicalModel for featurizing purposes. For example, CoreNLP Annotation objects can be
     * re-generated from the original source text.
     *
     * @param model the model, presumably with enough information encoded in its key-value store to recover the context
     *              object T
     * @return the context object T for this GraphicalModel, that will be cached and provided for future featurizing
     */
    protected abstract T restoreContextObjectFromModelTags(GraphicalModel model);

    /**
     * This is for subclasses, and is used to re-featurize before every training run, so that we can make sure that
     * changing feature sets are reflected in the next set of weights after retraining. It also means that we can save
     * logs of GraphicalModels that don't need to contain the features with them.
     *
     * @param model the model we'd like to featurize into
     * @param t the input element that will be used as context for the featurizing
     */
    protected abstract void featurizeModel(GraphicalModel model, T t);

    ////////////////////////////////////////////////////////////////////////
    // Private interfaces
    ////////////////////////////////////////////////////////////////////////

    /**
     * Checks, in a synchronized way, if a training run is currently going on any simple durable model that's in memory
     * on this machine.
     */
    private void launchTrainingRunIfNotRunning() {
        synchronized (this) {
            if (!trainingRunning) {
                trainingRunning = true;
            }
            else {
                return;
            }
        }

        // We only reach this if we are good to launch a training run

        Object thisClosure = this;
        new Thread(() -> {

            // Create the training set, and re-contextualize if necessary, and always re-featurize, in case the feature
            // set has changed since the last time we ran training.

            GraphicalModel[] frozenSet;
            frozenSet = new GraphicalModel[log.size()];
            for (int i = 0; i < frozenSet.length; i++) {
                frozenSet[i] = log.get(i);
            }
            for (GraphicalModel model : frozenSet) {
                // Generate context if necessary
                if (!context.containsKey(model)) {
                    context.put(model, restoreContextObjectFromModelTags(model));
                }
                // Refeaturize the model
                featurizeModel(model, context.get(model));
            }

            // Do the training

            AbstractBatchOptimizer optimizer = new BacktrackingAdaGradOptimizer();
            weights = optimizer.optimize(frozenSet, new LogLikelihoodDifferentiableFunction());

            // Write the results out to disk, just in case

            try {
                OutputStream os = new FileOutputStream(weightsPath);
                weights.writeToStream(os);

                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(namespacePath));
                oos.writeObject(namespace);
            } catch (IOException e) {
                e.printStackTrace();
            }

            // Check if we need to run again

            synchronized (thisClosure) {
                trainingRunning = false;
                if (log.size() > frozenSet.length) {
                    launchTrainingRunIfNotRunning();
                }
            }
        });
    }
}
