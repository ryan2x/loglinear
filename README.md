#loglinear
by Keenon Werling

A log linear model library with an intuitive interface, fast computation, and support for general tree-structure models.

Maven:

    <dependency>
        <groupId>com.github.keenon</groupId>
        <artifactId>loglinear</artifactId>
        <version>1.1</version>
    </dependency>

For documentation, see the "doc/" folder.

# What's new in v1.1:

This release was all about optimization. I provided another function in CliqueTree, calculateMarginalsJustSingletons(),
that skips a lot of the processing that's unnecessary if all you're looking for is individual variable marginals.

Most of the optimization is under the hood. If you mutate a PGM between calls to CliqueTree, CliqueTree will attempt to
re-use cached messages from previous rounds of message passing. This cuts down on overall computation by about 30%.
I also removed the boxing and unboxing in TableFactor, creating the special-case super-class NDArrayDoubles, to avoid
template forcing object-ness. There've also been a bunch of minor optimizations in CliqueTree.messagePassing(), not
worth mentioning by name.

The upshot: a huge boost in speed of inference, which is crucial for MCMC sampling applications like you find in my
paper, "On The Job Learning with Bayesian Decision Theory", [here](http://arxiv.org/pdf/1506.03140v1.pdf)