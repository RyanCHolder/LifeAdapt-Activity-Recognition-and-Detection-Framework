# LifeAdapt-Activity-Recognition-and-Detection-Framework
Comparison of HAR model performance when training on a variety of datasets and being tested on a novel dataset. 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Metrics](#metrics)
- [Citations](#citations)

## Introduction

This project uses 3 different machine learning models and compares their performance learning on several different HAR datasets then being tested on a novel dataset.
It is designed for users to be able to easily add their own models to compare against the models we used here. This project uses 7 datasets organized into a common formatting
from DAGHAR. The models are then trained on 6 of the seven datasets and tested on the seventh for all 7 possible combinations.

## Installation

To install and run this project, please follow all of the following steps (there is some significant modification to the models from their published versions):

1. Clone the repository

2. Download the dataset
    Follow this link and download the DAGHAR datasets: https://zenodo.org/records/11992126
    Put all of the dataset folders into one parent folder (the default name used in our code is DAGHAR)
3. Download models
    We used two HAR models and an additional basic 1 dimensional CNN for our experiment, follow these links to download the two HAR models:
    - https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate
    - https://github.com/sshao2013/ConvBoost
    - Clone the full repositories, place the folders into your working directory
        (keep the existing names, Attend-And-Discriminate and ConvBoost)

    A significant amount of modification to these models is necessary to be run on the DAGHAR datasets:
    - First, modifying Attend-And-Discriminate
        - Navigate to preprocess.py in the Attend-And-Discriminate folder
            - around line 56 you will find `x_train = contents["X_train"].astype(np.float32)` and other similar lines, for the "X_valid" lines, change valid to val
        - Then navigate to settings.py
            - At the beginning of the file add the following list: `DAGHAR_sets = ["KuHar", "MotionSense", "RealWorld_thigh","RealWorld_upperarm","RealWorld_waist","UCI"]`
            - At around line 25 you will see a variable called `choices` within the parser
                - Modify this line as follows: `choices=["opportunity", "skoda", "pamap2", "hospital","KuHar", "MotionSense", "RealWorld_thigh","RealWorld_upperarm","RealWorld_waist","UCI"],`
            - Around line 120 after the block headed by `elif args.dataset =="hospital" add the following block:
                ```python
                elif args.dataset in DAGHAR_sets:
                    args.num_class = 6
                    args.input_dim = 6
                    args.class_map = [
                        "sitting",
                        "standing",
                        "walking",
                        "climbing up",
                        "climbing down",
                        "running"
                    ]
            - Just after this you will see three paths `args.path_data` `args.path_raw` and `args.path_processed`
                - Replace each of these with the path to your datasets, the raw data is expecting .mat the result should be something like this:
                    `args.path_data = f"./Comb_Data/{args.dataset}.mat"`
                    `args.path_raw = f"./Comb_Data/{args.dataset}/raw"`
                    `args.path_processed = f"./Comb_Data/{args.dataset}/processed"`
                    - These paths assume Attend-And-Discriminate is in the same folder as main.py and Comb_Data
                - The Comb_Data and the folders within should be created by code in this model and the setup.py described in step 4
            - At around line 180, after the block `elif args.dataset == "hospital" add the following block
                ```python
                    elif args.dataset in DAGHAR_sets:
                        args.init_weights = "orthogonal"
                        args.beta = 0.3
                        args.dropout = 0.5
                        args.dropout_rnn = 0.25
                        args.dropout_cls = 0.5
        - Now Attend-And-Discriminate is ready to handle the DAGHAR data
    - Modifying ConvBoost
        - Navigate to ConvBoost/frame_wise/data/data_loader.py
            - add the following two lines to the top of the file
                - `DAGHAR_sets = ["KuHar", "MotionSense", "RealWorld_thigh","RealWorld_upperarm","RealWorld_waist","UCI"]`
                - `DAGHAR_ind = [4,5,6,7,8,9]`
            - In the HARDataLoader class __init__ function you will see a series of if statements, add the following after that block
                ```python
                if args.dataset in DAGHAR_ind:
                    file_name = DAGHAR_sets[args.dataset - 4]
                    self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y = loadingDB(args,
                        f'./Comb_Data/{file_name}',
                        6)
                    self.n_classes = 6
                    self.DB = 6
        - In the same directory go to data_preprocessing.py
            - You will see the loadingDB function with a series of if statements, add the following at the end of these (before the line ` X_train = X_train.astype(np.float32)`)
                ```python
                if DB == 6:
                    matfile = fileDir + '.mat'
                    data = scipy.io.loadmat(matfile)
                    X_train = data['X_train']
                    X_valid = data['X_val']
                    X_test = data['X_test']
                    y_train = data['y_train'].reshape(-1)
                    y_valid = data['y_val'].reshape(-1)
                    y_test = data['y_test'].reshape(-1)

                    #flatten the X data
                    X_train = np.array([x.flatten() for row in X_train for x in row]).reshape(X_train.shape[0], -1)
                    X_valid = np.array([x.flatten() for row in X_valid for x in row]).reshape(X_valid.shape[0], -1)
                    X_test = np.array([x.flatten() for row in X_test for x in row]).reshape(X_test.shape[0], -1)


                    mean_train = np.mean(X_train, axis=0)
                    std_train = np.std(X_train, axis=0)
                    X_train = (X_train - mean_train) / std_train
                    X_valid = (X_valid - mean_train) / std_train
                    X_test = (X_test - mean_train) / std_train

                    y_train = pd.get_dummies(y_train, prefix='labels')
                    y_valid = pd.get_dummies(y_valid, prefix='labels')
                    y_test = pd.get_dummies(y_test, prefix='labels')
        - Lastly and optionally we added an accuracy metric which was not originally displayed by ConvBoost. Navigate to ConvBoost/frame_wise/ensemble.py
            - In def show_results add the following lines
                - `current_acc = accuracy_score(truth_result, pred_result)`
                - `current_fused_acc = accuracy_score(truth_result, fused_pred)`
                - `print('curr_acc {:.3f} fused_acc {:.3f}'.format(current_acc, current_fused_acc))`

4. Run setup.py
    This project includes a file called setup.py which will create the file structure necessary to run our code and the two HAR models
    The necessary folders are included in the repository, but if you did not clone the entire repository, 
    then just run setup.py in your working directory

## Usage
After the code has all been setup, to reproduce our experiment, all that is necessary is to run main.py in your working directory.

If you modified the file structure at all you will want to include command line arguments --data_dir, --out_data_dir, and --output_dir
for the raw DAGHAR data, folder for the processed DAGHAR data, and the output of the runs respectively

To add your own model to the experiment, navigate into multi_run.py where you will see a list commands[command1, command2, command3]
Create your own command and add it to this list, which will then run your code. The commands are formatted to be command line calls since
this allows the already made HAR models to be run with minimal manipulation.

To format your command:
    - Create a list where each element represents one element of your command as a string
        - for example: ["python","mycode.py,","--argument1=value",""]
    - The multi_run.py code assumes that your code handles a --dataset command for which the value is just the name of the dataset (like KuHar)
        the empty string in your command is where this --dataset argument will be placed when running

## Metrics
An additional file called metrics.py calculates performance metrics of an HAR model for detection rather than recognition.
Explanation of how to use this can be found in the documentation of the file.
These metrics are not calculated by default for the three models we used.

The calculated metrics are mAP and IoU score:
    - mAP score or mean average precision score calculates the average precision at different IoU thresholds, then takes the mean of these averages
    - IoU is a measure of the overlap between the predicted windows of time for a certain activity and the true activity windows

## Citations
If you use this work, please cite the following

The two HAR models we used:

Attend-And-Discriminate
> Abedin, Alireza, Ehsanpour, Mahsa, Shi, Qinfeng, Rezatofighi, Hamid, and Ranasinghe, Damith C.  
> "Attend And Discriminate: Beyond the State-of-the-Art for Human Activity Recognition using Wearable Sensors."  
> *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*,  
> vol. 5, no. 1, 2021, doi: [10.1145/3448083](https://doi.org/10.1145/3448083).

ConvBoost
> Shao, Shuai, Guan, Yu, Zhai, Bing, Missier, Paolo, and Plötz, Thomas.  
> "ConvBoost: Boosting ConvNets for Sensor-based Activity Recognition."  
> *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*,  
> vol. 7, no. 2, June 2023, pp. 75, doi: [10.1145/3596234](https://doi.org/10.1145/3596234).  
> Association for Computing Machinery.
