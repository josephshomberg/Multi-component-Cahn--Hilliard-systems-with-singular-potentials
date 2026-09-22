# to fetch data from experiments and plot the data

# load libraries
import matplotlib.pyplot as plt
import numpy as np
import pylab as pylab
import csv
#import os

# data locations
filename8  = 'q=0.8/data/energy[n].csv'
filename9  = 'q=0.9/data/energy[n].csv'
filename10 = 'q=1.0/data/energy[n].csv'
filename11 = 'q=1.1/data/energy[n].csv'
filename12 = 'q=1.2/data/energy[n].csv'

DATA = 6  # this pulls from colum 6 of 0-7
row_begin = 0
row_end   = 50000  # rows are from 0-50,000

# for storing values
y8 = []
y9 = []
y10 = []
y11 = []
y12 = []

# read data
with open(filename8,'r') as file8:
    lines = csv.reader(file8, delimiter=',')
    row_count = 1
    for row in lines:
        row_count += 1
        if row_count <= row_begin:
            continue
        y8.append(float(row[DATA]) + 1)

with open(filename9,'r') as file9:
    lines = csv.reader(file9, delimiter=',')
    row_count = 1
    for row in lines:
        row_count += 1
        if row_count <= row_begin:
            continue
        y9.append(float(row[DATA]) + 1)

with open(filename10,'r') as file10:
    lines = csv.reader(file10, delimiter=',')
    row_count = 1
    for row in lines:
        row_count += 1
        if row_count <= row_begin:
            continue
        y10.append(float(row[DATA]) + 1)

with open(filename11,'r') as file11:
    lines = csv.reader(file11, delimiter=',')
    row_count = 1
    for row in lines:
        row_count += 1
        if row_count <= row_begin:
            continue
        y11.append(float(row[DATA]) + 1)

with open(filename12,'r') as file12:
    lines = csv.reader(file12, delimiter=',')
    row_count = 1
    for row in lines:
        row_count += 1
        if row_count <= row_begin:
            continue
        y12.append(float(row[DATA]) + 1)

# plot data from the file
x = np.arange(row_end + 1)
plt.clf()
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, y8, label='q = 0.8')
ax.plot(x, y9, label='q = 0.9')
ax.plot(x, y10, label='q = 1.0')
ax.plot(x, y11, label='q = 1.1')
ax.plot(x, y12, label='q = 1.2')

#ax.set_xlabel('Iteration (n)')
#ax.set_ylabel('Energy')
#ax.set_title('Tsallis Energy')
ax.legend(loc='upper right', shadow=True)
ax.grid(True)

ax.set_ylim(0.4, 1.7)

plt.tight_layout()
plt.savefig('Tsallis_energy.png', dpi=300)
#plt.show()
