"""
This is the example code that will be used in the full system.
Not a full functioning code but the parts for the channel to be put in
the full structured system.

The configuration part should be put at the beginning of the full code
after the path setup and the imports to easily change the parameters.
"""
#required imports for the channel
import numpy as np
from fyp_channel.channel import CompositeChannel, SSFMChannel, CDCompensation

def get_configuration():
    """It would be better to have all the parameters in here not just
    channel to make the code more structured and easier to edit.
    """

    config = {

        'fiber_length': 400.0,  
        'span_length': 80.0,  
        'step_size': 0.5,
        'alpha': 0.2,
        'D': 16,
        'gamma': 1.3,
        'Fc': 193.1e12,
        'NF': 4.5,
        'seed': 10,

    }

    return config


def channel(sigTx, Fs, config):
    #extract parameters from config
    fiber_length = config['fiber_length']
    span_length = config['span_length']
    step_size = config['step_size']
    alpha = config['alpha']
    D = config['D']
    gamma = config['gamma']
    Fc = config['Fc']
    NF = config['NF']
    seed = config['seed']

    #calculate number of spans
    n_spans = int(np.ceil(fiber_length / span_length))

    #create channel
    ch = CompositeChannel()

    #configure and add ssfm channel
    ssfm_channel = SSFMChannel(
        Ltotal = fiber_length,
        Lspan = span_length,
        hz = step_size,
        alpha = alpha,
        D = D,
        gamma = gamma,
        Fc = Fc,
        amp = 'edfa',
        NF = NF,
        seed = seed,
        prgsBar = False,
    )

    ch.add_effect(ssfm_channel)

    #configure and add CD compensation
    cd_comp = CDCompensation(
        L = fiber_length,
        D = D,
        Fc = Fc,
    )

    ch.add_effect(cd_comp)

    #propagate signal through channel
    sigRx = ch.propagate(sigTx, Fs)

    return sigRx