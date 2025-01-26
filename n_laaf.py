import tensorflow as tf

class NLLAFLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='tanh',
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        """
        L-LAAF Layer.
        Args:
            units: Number of units in the layer.
            activation: Base activation function to use (default: 'relu').
            Additional arguments for the dense layer (e.g., use_bias, kernel_initializer, etc.).
        """
        super(NLLAFLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.a_k = None  # Learnable parameter
        self.linear_layer = None  # Linear transformation

    def build(self, input_shape):
        """
        Initialize the layer's parameters.
        """
        # Initialize the dense (linear transformation) layer
        self.linear_layer = tf.keras.layers.Dense(
            units=self.units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )

        # Single learnable parameter for each neuron
        self.a_k = self.add_weight(
            shape=(self.units, ),  # Scalar for the layer
            initializer='ones',  # Initialize a_k to 1
            trainable=True,
            name='a_k'
        )

    def call(self, inputs):
        """
        Forward pass.
        """
        # Apply the linear transformation
        linear_output = self.linear_layer(inputs)

        # Scale the entire layer's output with a_k
        adaptive_output = self.activation(self.a_k * linear_output)

        return adaptive_output
