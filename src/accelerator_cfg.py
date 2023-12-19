from enum import Enum
from collections import namedtuple


class AcceleratorType(Enum):
    Eyeriss = 1


EyerissAcceleratorState = namedtuple("EyerissAcceleratorState",
                                     ['pe_array_x', 'pe_array_y',
                                      'precision', 'sram_size',
                                      'ifmap_spad_size', 'weights_spad_size', 'psum_spad_size'])


class AcceleratorProfile:
    """General characteristics from the given accelerator
    """
    def __init__(self, accelerator_type):
        self.type = accelerator_type

        if accelerator_type == AcceleratorType.Eyeriss:
            self.state = EyerissAcceleratorState

            # accelerator parameters, according to https://ieeexplore.ieee.org/document/7738524
            self.width = 14
            self.height = 12
            self.num_pes = self.width * self.height
            self.precision_weights = 16                 # in bits, fixed-point
            self.precision_activations = 16             # in bits, fixed-point
            self.glb_sram_size = 108000                 # in bytes
            self.ifmap_spad_size = 24                   # in bytes (12b x 16b)
            self.weights_spad_size = 448                # in bytes (224b x 16b)
            self.psum_spad_size = 48                    # in bytes (24b x 16b)
            self.dataflow = 'row_stationary'
            self.num_banks = 25
            self.sram_per_bank = 4000                   # in bytes (512b x 64b)
            self.clock_rate = 250 * 1e6                 # in Hz (100 - 250 MHz allowed)
            self.dram_precision = 64                    # in bits

            # design space parameters
            width_options = height_options = [8, 10, 12, 14, 16, 20, 25]
            ifmap_spad_size_options = [8, 12, 16, 24, 32, 40, 48, 64]
            weights_spad_size_options = [256, 320, 384, 448, 512, 576, 640]
            psum_spad_size_options = [16, 24, 32, 40, 48, 64, 80, 96]
            sram_size_options = [45000, 60000, 800000, 108000, 120000, 140000, 180000]
            precision_options = [8, 16, 32]
            # prepare dictionary with design space parameters, according to the self.state class
            self.design_space = {
                'pe_array_x': width_options,
                'pe_array_y': height_options,
                'precision': precision_options,
                'sram_size': sram_size_options,
                'ifmap_spad_size': ifmap_spad_size_options,
                'weights_spad_size': weights_spad_size_options,
                'psum_spad_size': psum_spad_size_options
            }

        else:
            raise NotImplementedError("Accelerator types other than Eyeriss-like are not supported")

    def __repr__(self) -> str:
        if self.type == AcceleratorType.Eyeriss:
            return f'Accel{self.precision_weights}b'
        else:
            return super().__repr__()
