class Constants:
    BREAST_MNIST = 'breast_mnist'
    PNEUMONIA_MNIST = 'pneumonia_mnist'
    RETINA_MNIST = 'retina_mnist'
    OCT_MNIST = 'oct_mnist'
    SUPPORTED_DATASETS = [BREAST_MNIST, PNEUMONIA_MNIST, RETINA_MNIST, OCT_MNIST]

    PROTOTYPE_COEFF = 0.001
    N_PROTOTYPES_PER_CLASS = 25