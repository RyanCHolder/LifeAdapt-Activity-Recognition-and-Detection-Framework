
import pandas as pd
import numpy as np
import os
import scipy.io
import torch
import random

"""Reformat data according to the models used for this experiment
    Data is formatted with each row as one time stamp (one second)
    Each row then has 6 values, acceleration and gyroscope x y and z
    Labels are numbers from 0 to 5
    
    The DAGHAR data is originally split into train, validate, and test
    This code combines them, and then creates 7 files with a new split where
    training and validation data are taken from 6 of the 7 files then the 7th file is used for testing"""

def reformat(df):
    # reformats a dataframe according to what the models expect
    # the format should be one row per timestamp, each feature is a separate column
    # returns the reformatted data as two separate arrays for x and y
    arrays = []
    for index, row in df.iterrows():
        #get each 60 second period of data from each of the 6 features
        sets = [row[i*60:(i+1)*60].values for i in range(6)]
        
        # Stack them horizontally
        result = np.column_stack(sets)
        arrays.append(result)

    #stack the arrays
    X_data = np.vstack(arrays)

    #get the labels
    labels = df['standard activity code']
    # extend the labels to match the new formatting
    # DAGHAR originally has one label per 60 second interval, so we want 60 labels, one for each second
    y_data = np.repeat(labels, 60)

    return X_data, y_data

def create_combined(dir):
    print("Combining data from premade splits")
    # Iterate over each folder
    for folder_name in os.listdir(dir):
        folder_path = os.path.join(dir, folder_name)
        
        # Initialize an empty list to store dataframes from each CSV file
        dataframes = []
        
        if os.path.exists(os.path.join(folder_path,"combined.csv")):
            print(f"Combined data already created for {folder_name}")
        
        else:
            # Iterate over CSV files in the current folder
            for csv_file in os.listdir(folder_path):
                if csv_file.endswith('.csv'):
                    csv_file_path = os.path.join(folder_path, csv_file)
            
                    # Read the CSV file and append to the list of dataframes
                    df = pd.read_csv(csv_file_path)
                    dataframes.append(df)
            
            # Vertically stack all dataframes for the current folder
            combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
            
            # Save the combined dataframe
            combined_df.to_csv(os.path.join(folder_path, "combined.csv"), index=False)
    print("Data combining completed")


def process(data_dir="DAGHAR",out_dir="Comb_Data",format=0):

    #combine the splits of the DAGHAR datasets into one csv
    create_combined(data_dir)

    #format: 1 is for .mat, 2 is for .npz, 0 is for both

    for folder in os.listdir(data_dir):
        #folder for testing data
        folder_path = os.path.join(data_dir, folder)
        csv_path = os.path.join(folder_path, 'combined.csv')

        X_train = []
        y_train = []
        X_val = []
        y_val = []

        #get testing data first
        df = pd.read_csv(csv_path)
        X_test, y_test = reformat(df)

        #combine other four folders for training and validation data
        for folder2 in os.listdir(data_dir):
            folder2_path = os.path.join(data_dir, folder2)

            if folder2_path != folder_path:
                #folder for training data
                csv2_path = os.path.join(folder2_path, 'combined.csv')
                #format data
                df = pd.read_csv(csv2_path)
                x, y = reformat(df)

                # separate into training and validation

                # Calculate the index to split at for 90/10 train/validation
                split_index = int(0.9 * len(x))

                # Split the array and append
                X_train.append(x[:split_index,:])
                y_train.append(y[:split_index])
                X_val.append(x[split_index:,:])
                y_val.append(y[split_index:])

        # stack the training and validation arrays
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        X_val = np.vstack(X_val)
        y_val = np.concatenate(y_val)

        # create dictionary
        data_dict = {
            'X_train':X_train,
            'y_train':y_train,
            'X_val':X_val,
            'y_val':y_val,
            'X_test':X_test,
            'y_test':y_test
        }

        #test set name
        test_name = os.path.basename(folder)
        print("Saving Processed data with " + test_name + " as testing data...")

        #save results to .mat and or .npz file
        if format == 1:
            #save dictionary to .mat, so the name of the file indicates which dataset is used for testing
            scipy.io.savemat(out_dir+"/"+folder+'.mat', data_dict)
        elif format == 2:
            #save .npz version
            np.savez(out_dir+"/"+folder+".npz", **data_dict)
        elif format == 0:
            scipy.io.savemat(out_dir+"/"+folder+'.mat', data_dict)
            np.savez(out_dir+"/"+folder+".npz", **data_dict)
    
        print("Saved")  

if __name__== "__main__":
    process()


