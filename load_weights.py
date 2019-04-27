import torch

def load_weights(path):
    state = torch.load(path)

    # shared weights start with 'module.' since DataParallel was used during training
    # Parallel isn't used here and therefore prefix 'module.' need to be deleted
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
    state_model = state['model']
    def fix_name(name):
        if name.startswith('module.'):
            return name[len('module.'):]
        return name
    converted = {fix_name(k): state_model[k] for k in state_model}
    state['model'] = converted
    return state
