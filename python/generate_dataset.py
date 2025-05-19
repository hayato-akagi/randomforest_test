import csv
import random

# Settings
NUM_TRAIN = 500
NUM_TEST = 200
TRAIN_FILE = '/data/input.csv'

# Header
headers = ['magnetic_field', 'electric_field', 'position', 'momentum', 'label']

def generate_sample():
    magnetic_field = random.uniform(-1.0, 1.0)      # Tesla
    electric_field = random.uniform(0, 300)         # V/m
    position = random.uniform(-10.0, 10.0)          # Arbitrary units
    momentum = random.uniform(-2.0, 2.0)            # Arbitrary units

    # Heuristic for labeling (simulating physics):
    label = 1 if (magnetic_field * momentum > 0.05 and electric_field > 150) else 0

    return [round(magnetic_field, 3), round(electric_field, 2),
            round(position, 2), round(momentum, 3), label]

def generate_balanced_data(n):
    half_n = n // 2
    data = []
    count = {0: 0, 1: 0}
    while count[0] < half_n or count[1] < half_n:
        sample = generate_sample()
        label = sample[-1]
        if count[label] < half_n:
            data.append(sample)
            count[label] += 1
    return data

def write_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"{filename} written with {len(data)} samples (50:50).")

if __name__ == "__main__":
    train_data = generate_balanced_data(NUM_TRAIN)

    write_csv(TRAIN_FILE, train_data)
