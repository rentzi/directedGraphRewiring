import pickle
import os
import numpy as np

# Utility functions


def save_var(var, file_path):
    '''
     save_var saves a var to a specified filePath
     INPUT:
     var: the variable to be saved

     file_path: the filepath you want to save the var, for example data/.../var.pckl. If the path does not exist, it creates it
    '''

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):  # makes the directory if it does not exist
        os.makedirs(directory)

    # uses pickle to serialize and save the variable var
    pickle_out = open(file_path, 'wb')
    pickle.dump(var, pickle_out)
    pickle_out.close()


def load_var(file_path):
    '''
    LOADVAR loads a var from a specified filePath
    INPUT:
    filePath: where the variable is
    OUTPUT:
    var: the variable loaded
    '''

    pickle_in = open(file_path, 'rb')
    var = pickle.load(pickle_in)
    pickle_in.close()

    return var


def not_exist(arg1, arg2):
    """
    checks if arguments exist in workspace

    args

        arg1: str
        arg2: str

    returns

        True or False
    """
    if (arg1 not in locals()) | (arg2 not in locals()):
        return True


def check_is_null_exception(arg1, A, tau, pRandRewire):
    """
    Check whether an argument arg1 is empty or not

    args

        arg1: ?
            the argument that is checked
        A:
            Adjacency matrix
        tau: ?
            heat dispersion parameter
        pRandRewire: ?
            probability of random rewiring

    returns

        A: ?
    """
    if (len(arg1) == 0):
        print('For tau = %f, and p(rand) = %f,the graph is problematic' % (tau, pRandRewire))
        return A
