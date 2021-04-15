import sys
from SceneRecognition.Execution import *
if __name__ == '__main__':
    try:
        dataset_path = sys.argv[1]
        execute(dataset_path)
    except Exception as e:
        print(e)
        print("Wrong number of arguments")