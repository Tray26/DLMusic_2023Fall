import os
import csv

csv_path = './support_data/singers.csv'


with open(csv_path, newline='') as readfile:
    rows = csv.reader(readfile)
    rows = list(rows)
    # print(rows.shape)
