from .DDUDSU import DDUDSU
def model_bulider(config):
    model_type = config['model_type']
    embeading_dim = config['embeading_dim']
    stage = config['stage']
    type = config['type']
    if model_type == 'DDUDSU':
        model = DDUDSU(embeading_dim,stage)

    return model