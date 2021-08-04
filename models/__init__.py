import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.

    model_filename = "models." + model_name.lower() + "_model"
    print('model_filename is',model_filename)
    modellib = importlib.import_module(model_filename)
    print('modellib is ',modellib)
    # you will get result.
    # model_filename is models.dcomp_model
    # modellib is  <module 'models.dcomp_model' from '/home/rizhao/test/ACMNet/models/dcomp_model.py'>

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    print('target_model_name is ',target_model_name)
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    model = find_model_using_name(opt.model)
    print('after find_model_using_name, model is',model)
    instance = model()
    print('in the create model, instance is',instance)
    instance.initialize(opt)
    instance.print_networks()
    print("model [%s] was created" % (instance.name()))
    return instance
