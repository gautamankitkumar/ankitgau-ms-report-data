from ase.db import connect
import random

def cross_split(dbname, k):
    """k: k-fold cross"""
    db = connect(dbname + '.db')
    data = list(db.select())
    N = len(data)
    n_val = N // k
    for i in range(k):
        val_db = connect(dbname + f'-valid-{i}.db')
        tra_db = connect(dbname + f'-train-{i}.db')
        val_data = data[i*n_val:(i+1)*n_val]
        tra_data = data[:i*n_val] + data[(i+1)*n_val:]
        for entry in val_data:
            val_db.write(entry)
        for entry in tra_data:
            tra_db.write(entry)

        print(val_db.count())
        print(tra_db.count())


if __name__ == "__main__":
    cross_split('./bulk-dfts', 4)