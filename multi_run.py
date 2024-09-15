import os

"""Handles running of the different models using system commands.
    Functionality for additional models can be added by appending a new command to the commands list
    Also handles saving results to text files in the output folder"""


# list of dataset names
datasets = ["KuHar", "MotionSense", "RealWorld_thigh","RealWorld_upperarm","RealWorld_waist","UCI","WISDM"]

#the empty argument will be replaced with the dataset keyword argument (there are some spaces to make the command call work properly)
command1 = ["python ","Attend-And-Discriminate/main.py ","", " --train_mode"] # run attend and discriminate model
command2 = ["python ","ConvBoost/frame_wise/main.py ","", " --model_name=CNN-3L"] # run ConvBoost model
command3 = ["python ","1DCNN.py ",""] # run 1D CNN

# condense into one list
commands = [command1,command2,command3]
# ids for file writing later
command_ids = ["Attend And Discriminate", "ConvBoost","1DCNN"]

def run(outdir="output"):
    for i in range(len(datasets)):
        for j in range(len(commands)):
            if j == 1:
                # ConvBoost expects an integer, not a name, where 1,2, and 3 are the original datasets from their project
                commands[j][2] = f"--dataset={i+4}"
            else:
                commands[j][2] = f"--dataset={datasets[i]}"

        # create output file for this dataset
        out_file = os.path.join(outdir,f"{datasets[i]}_results.txt")

        print(f"Training models to test on {datasets[i]}")

        #dataset header in output file
        with open(out_file, "w") as f:  
                f.write(f'Testing on {datasets[i]} dataset\n\n')

        # iterate through each model
        for j in range(3):
            #model headers in output file
            with open(out_file, "a") as f:  
                f.write(f'\n\nResults of testing using {command_ids[j]} model:\n\n')

            #progress update
            print(f"Running {command_ids[j]}")
            #turn the command into a single string
            c = ''.join(commands[j])
            #This set command is necessary if running on windows
            os.system(f'set PYTHONIOENCODING=UTF-8 && {c} >> {out_file}')
            print(f"Finished run")

        print("Results saved \n")