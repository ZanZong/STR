import obj_dict_test

def gradients(ys, xs, grad_ys=None, checkpoints='collection', **kwargs):
    print("a mocked gradient function with args!!")

obj_dict_test.__dict__['gradients'] = gradients

obj_dict_test.gradients(1,2)