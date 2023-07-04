from .arch import Net


def model_bulider(config):
    model_type = config['model_type']
    embeading_dim = config['embeading_dim']
    stage = config['stage']
    if model_type == 'Net':
        model = Net(embeading_dim,stage)
    return model