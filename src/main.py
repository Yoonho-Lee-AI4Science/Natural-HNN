import numpy as np
import utils
from trainer import Model_Trainer 
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    args = utils.parse_args()
    args.root_dir = ROOT_DIR
    args.code_dir = ROOT_DIR + "/src"
    args.data_dir = ROOT_DIR + "/dataset/"
    mt = Model_Trainer(args)
    mt.train()
    mt._train_writer.close()
    mt._val_writer.close()

if __name__ == "__main__":
    main()