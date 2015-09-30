#loglinear
by Keenon Werling

A log linear model library with an intuitive interface, fast computation, and support for general tree-structure models.

Maven:

    <dependency>
        <groupId>com.github.keenon</groupId>
        <artifactId>loglinear</artifactId>
        <version>1.1.2</version>
    </dependency>

For documentation, see the "doc/" folder.

# What's new in v1.1: Inference Speed

This release was all about optimization. I provided another function in CliqueTree, calculateMarginalsJustSingletons(),
that skips a lot of the processing that's unnecessary if all you're looking for is individual variable marginals.

Most of the optimization is under the hood. If you mutate a PGM between calls to CliqueTree, CliqueTree will attempt to
re-use cached messages from previous rounds of message passing. This cuts down on overall computation by about 30%.
I also removed the boxing and unboxing in TableFactor, creating the special-case super-class NDArrayDoubles, to avoid
template forcing object-ness. There've also been a bunch of minor optimizations in CliqueTree.messagePassing(), not
worth mentioning by name.

The upshot: a huge boost in speed of inference, which is crucial for MCMC sampling applications like you find in my
paper, "On The Job Learning with Bayesian Decision Theory", [here](http://arxiv.org/pdf/1506.03140v1.pdf)

# What's new in v1.1.1: model.getVariableSizes()

I added a quick convenience function to get an int[] array, as a snapshot of the sizes of the variables given by the
factors of the model.

# What's new in v1.1.2: CliqueTree cacheing bugfix

There were some dangerous assumptions in the cacheing of messages in CliqueTree, and if you ran with asserts enabled,
some aggressive modifications of the model could cause your CliqueTree to crash with an assert. This is now fixed.

# Coming up in v1.2: Distributed Learning

Distributed optimization for learning! Since the raw gradient computation is unlikely to get much faster after all this
optimization in v1.1, it's time to make sure that learning can be handled by a cluster of machines accessible over the 
network.
