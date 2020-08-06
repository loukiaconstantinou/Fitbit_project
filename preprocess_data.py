#import libraries
import pandas as pd
import os
from functools import reduce
import ast
import numpy as np

#parser function
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Input data path", required=True)
    parser.add_argument("--name", type=str, default="Loukia", 
    help="person to import data from", choices=["Loukia", "Kyriacos", "Irene", "Christina"])
    args = parser.parse_args()
    return args

#gets the correct path depending on which user's data we want to analyse
if __name__ == "__main__":
    args = parse_args()
    folderpath = args.datapath
    if args.name == 'Loukia':
        folderpath = r"{}/LoukiaConstantinou/user-site-export".format(folderpath)
    elif args.name == 'Kyriacos':
        folderpath = r"{}/KyriacosXanthos/user-site-export".format(folderpath)
    elif args.name == 'Irene':
        folderpath = r"{}/IreneConstantinou/user-site-export".format(folderpath)   
    elif args.name == 'Christina':
        folderpath = r"{}/Christina/user-site-export".format(folderpath) 


    if not os.path.isdir("excel_files_{}".format(args.name)):
        os.makedirs("excel_files_{}".format(args.name))

    outdir = "excel_files_{}".format(args.name)

    filepaths =[]
    for item in os.listdir(folderpath):
        if not item.startswith('.') and os.path.isfile(os.path.join(folderpath, item)):
            new_file = os.path.join(folderpath, item)
            filepaths.append(new_file)
    
    #split the path name twice to get the unique name to name our excel files
    for path in filepaths:
            xl_name = path.split(folderpath)
            xl_name2 = xl_name[1].split('.')
            pd.read_json("{}".format(path)).to_excel("{}/{}.xlsx".format(outdir, xl_name2[0]))

    folderpath2 = r"{}".format(outdir) #make sure to put the 'r' in front
    filepaths2  = [os.path.join(folderpath2, name) for name in os.listdir(folderpath2)]

    if not os.path.isdir("combined_csv_files_{}".format(args.name)):
        os.makedirs("combined_csv_files_{}".format(args.name))

    outdir2 = "combined_csv_files_{}".format(args.name)

    dictionary = {}  
    for path in filepaths2:  
        key = os.path.basename(path).split('-')[0]
        group = dictionary.get(key,[])
        group.append(path)  
        dictionary[key] = group

        
    for key in dictionary:
        excel_names = dictionary[key]
        #read the excel files
        excels = [pd.ExcelFile(name) for name in excel_names]
        #turn them into dataframes
        frames = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels]
        #delete the first row for all frames except the first (i.e. remove the header row - assumes it's the first)
        frames[1:] = [df[1:] for df in frames[1:]]
        #concatenate all the frames
        combined = pd.concat(frames)
        # write it out into csv 
        combined.to_csv("{}/combined_{}.csv".format(outdir2, key), header=False, index=False)
