import csv

def get_db():
    db = []
    with open('db.csv') as fh:
        for line in fh.readlines():
            db.append(line.strip())

    return db
