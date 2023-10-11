import os
import csv

def get_singers(singers_path='./support_data/singers.csv'):
    with open(singers_path, newline='') as readsingers:
        singers_list = csv.reader(readsingers)
        singers_list = list(singers_list)
        singers_list = singers_list[0]
    return singers_list

if __name__ == "__main__":
    # csv_path = './support_data/singers.csv'
    singers_list = get_singers()
    print(singers_list)
