class ODEop(theano.tensor.Op):
    def __init__(self, state, numpy_vsp):
        self._state = state

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)

        return theano.tensor.Apply(self, [x], [x.type()])

    def perform(self, node, inputs_storage, output_storage):
        x = inputs_storage[0]
        out = output_storage[0]

        out[0] = self._state(x)  # get the numerical solution of ODE states

    # def grad(self, inputs, output_grads):
    #     x = inputs[0]
    #     g = output_grads[0]

    #     grad_op = ODEGradop(self._numpy_vsp)  # pass the VSP when asked for gradient
    #     grad_op_apply = grad_op(x, g)

    #     return [grad_op_apply]