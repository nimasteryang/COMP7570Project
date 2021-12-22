# package
1. pytorch
2. pytorch-geometric
3. tqdm
4. pandas
5. numpy
6. sklearn
7. matplotlib
8. seaborn
9. tensorboardX

# run
1. `python main.py`
    1. random seed can be change inside `main.py`

# complete pipeline from SC to graph to experiment
1. unzip `data/contract.zip`, it will create a folder name `contract`
2. run preprocessing.py, it will analysis the SC and output pyg data
3. run data_split.py it will split the pyg data into train, valid(for fine tune), and test
4. run main.py 