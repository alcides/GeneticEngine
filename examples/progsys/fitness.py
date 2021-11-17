import os
import ast
import sys
sys.path.insert(1, 'C:\\Users\\leoni\\Desktop\\Master\\Scriptie\\GeneticEngine')

CWD = os.path.dirname(os.path.realpath(__file__))
FILE_NAME = "Number IO"
DATA_FILE_TRAIN = "C:\\Users\\leoni\\Desktop\\Master\\Scriptie\\GeneticEngine\\examples\\progsys\\data\\{}\\Train.txt".format(FILE_NAME)
DATA_FILE_TEST = "C:\\Users\\leoni\\Desktop\\Master\\Scriptie\\GeneticEngine\\examples\\progsys\\data\\{}\\Test.txt".format(FILE_NAME)

def bla():
    return "print(1 + 1)\nprint(2.5 * 2)"

def get_data(data_file_train,data_file_test):
    with open(data_file_train, 'r') as train_file, \
            open(data_file_test, 'r') as test_file:
        train_data = train_file.read()
        test_data = test_file.read()

    t = train_data.split('\n')

    inval = t[0].strip('inval = ')
    outval = t[1].strip('outval = ')
    inval = ast.literal_eval(inval)
    outval = ast.literal_eval(outval)
    return inval,outval


inval,outval = get_data(DATA_FILE_TRAIN,DATA_FILE_TEST)
imported = __import__(FILE_NAME + "-Embed")

evolved_function = lambda x,y: x+y

fitness, error, cases = imported.fitness(inval,outval,evolved_function)
print(fitness)
print(all(cases))


