from regularizer import Regularizer, SigmoidRegularizer
import torch.nn as nn 

def get_regularizer(regularizer_setting, entity_dim, neg_input_possible=True, entity=False):
    """
    :param neg_input_possible: for matrix_L1 (class MatrixSumRegularizer)
    :param dual: only apply regularizer to the first half embeddings (after chunk dim=-1) (for sigmoid only)
    """
    if entity:
        key = 'e_reg_type'
    else:
        key = 'type'
    if regularizer_setting[key] == '01':
        regularizer = Regularizer(base_add=0, min_val=0, max_val=1)
    elif regularizer_setting[key] == 'sigmoid':
        regularizer = SigmoidRegularizer(entity_dim, dual=regularizer_setting['dual'])
    elif regularizer_setting[key] == "softmax":
        regularizer = nn.Softmax(dim=-1)
    return regularizer
