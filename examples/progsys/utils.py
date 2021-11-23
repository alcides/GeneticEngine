import os
import ast
import importlib as il




def get_data(data_file_train,data_file_test,test=False):
    with open(data_file_train, 'r') as train_file, \
            open(data_file_test, 'r') as test_file:
        train_data = train_file.read()
        test_data = test_file.read()

    if test:
        t = test_data.split('\n')
    else:
        t = train_data.split('\n')

    inval = t[0].strip('inval = ')
    outval = t[1].strip('outval = ')
    inval = ast.literal_eval(inval)
    outval = ast.literal_eval(outval)
    return inval, outval


def import_embedded(FILE_NAME):
    imported = il.import_module(f"embed.{FILE_NAME}")
    return imported
