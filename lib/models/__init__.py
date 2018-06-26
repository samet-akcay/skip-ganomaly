##
import importlib

##
def load_model(opt, dataloader):
    """ Load model based on the model name.

    Arguments:
        opt {[argparse.Namespace]} -- options
        dataloader {[dict]} -- dataloader class

    Returns:
        [model] -- Returned model
    """
    model_name = opt.model
    model_path = f"lib.models.{model_name}"
    model_lib  = importlib.import_module(model_path)
    model = getattr(model_lib, model_name.title())
    return model(opt, dataloader)