# Triple GANs

Deep Generative models(DGM) are neural networks which tries to estimate the
likelihood of each observation and to create new samples from the underlying
distribution of data, Synthetic data generation is a specific application of deep
generative models where the goal is to create realistic and novel data samples.
GANs, in particular, consist of a generator network that produces new samples
and a discriminator network that distinguishes between real and generated samples.
Through an adversarial training process, both networks improve iteratively, with
the generator learning to produce more realistic data.


However, challenges exist. Ensuring diversity and realism in generated samples
is an ongoing research focus. Issues like training stability, mode collapse, and
addressing biases in generated data are important considerations. Ethical concerns
regarding the potential misuse of synthetic data and privacy implications must be
taken into account. The Triple GAN focuses on classification and class-conditional
generation of data samples and also a better framework to work with as it requires
less number of epochs to train, can perform well even without data argumentation
and can perform semi-supervised learning(SSL) tasks better.
