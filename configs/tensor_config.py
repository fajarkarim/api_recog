import os

current_path = os.getcwd()
base_path = os.path.dirname(current_path + '/configs')

def path():
    tensor_path = {
    'model_path': base_path + '/tensor_model' + '/retrained_graph.pb',
    'label_path': base_path + '/tensor_model' + '/retrained_labels.txt'
    }
    return tensor_path
