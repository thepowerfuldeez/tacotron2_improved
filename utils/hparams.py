import yaml
from pathlib import Path


class AttrDict(dict):
    """
    A class to convert a nested Dictionary into an object with key-values
    that are accessible using attribute notation (AttrDict.attribute) instead of
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse down nested dicts (like: AttrDict.attr.attr)
    """

    # Inspired by:
    # http://stackoverflow.com/a/14620633/1551810
    # http://databio.org/posts/python_AttributeDict.html

    def __init__(self, iterable, **kwargs):
        super(AttrDict, self).__init__(iterable, **kwargs)
        for key, value in iterable.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            else:
                self.__dict__[key] = value


def create_hparams(config_file="config_singlespeaker.yaml"):
    hparams = AttrDict(yaml.safe_load(Path(config_file).open()))
    return hparams
