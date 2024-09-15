import argparse

from preprocess import process
from multi_run import run

"""Central script to both preprocess raw data from DAGHAR and run the experiment from all three different models"""

def main():
    #get command line argument
    parser = argparse.ArgumentParser(description="Specifying file paths")

    parser.add_argument("--data_dir",type=str,default="DAGHAR",help="Directory for DAGHAR raw csv data")
    parser.add_argument("--out_data_dir",type=str,default="Comb_Data",help="Directory for output of processed data")
    parser.add_argument("--output_dir",type=str,default="output",help="Folder for results text files")

    args = parser.parse_args()

    #run the preprocess pipeline
    process(args.data_dir,args.out_data_dir)

    print("Data Preprocessing Completed, Beginning Experiment")

    #run the three models
    run(args.output_dir)

if __name__ == '__main__':
    main()