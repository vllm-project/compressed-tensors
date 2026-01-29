from compressed_tensors.transform import HadamardFactory, TransformFactory
from compressed_tensors.transform.utils.hadamard import random_hadamard_matrix
from torch import device, dtype
from torch.nn import Parameter


@TransformFactory.register("random-hadamard")
class RandomHadamardFactory(HadamardFactory):
    """
    Factory used to apply random hadamard transforms to a model

    :param name: name associated with transform scheme
    :param scheme: transform scheme which defines how transforms should be created
    :param seed: random seed used to transform weight randomization
    """

    def _create_weight(
        self,
        size: int,
        device: device,
        construct_device: device,
        precision: dtype,
    ) -> Parameter:
        data = random_hadamard_matrix(size, precision, construct_device, self.generator)
        data = data.to(device=device)
        return Parameter(data, requires_grad=self.scheme.requires_grad)
