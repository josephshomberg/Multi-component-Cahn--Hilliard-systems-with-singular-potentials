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

DATA = 7  # this pulls from colum 7 of 0-7
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
        y8.append(float(row[DATA]))

with open(filename9,'r') as file9:
    lines = csv.reader(file9, delimiter=',')
    row_count = 1
    for row in lines:
        row_count += 1
        if row_count <= row_begin:
            continue
        y9.append(float(row[DATA]))

with open(filename10,'r') as file10:
    lines = csv.reader(file10, delimiter=',')
    row_count = 1
    for row in lines:
        row_count += 1
        if row_count <= row_begin:
            continue
        y10.append(float(row[DATA]))

with open(filename11,'r') as file11:
    lines = csv.reader(file11, delimiter=',')
    row_count = 1
    for row in lines:
        row_count += 1
        if row_count <= row_begin:
            continue
        y11.append(float(row[DATA]))

with open(filename12,'r') as file12:
    lines = csv.reader(file12, delimiter=',')
    row_count = 1
    for row in lines:
        row_count += 1
        if row_count <= row_begin:
            continue
        y12.append(float(row[DATA]))

# plot data
print("q=0.8:", len(y8))
print("q=0.9:", len(y9))
print("q=1.0:", len(y10))
print("q=1.1:", len(y11))
print("q=1.2:", len(y12))

plt.clf()
fig, ax = plt.subplots(figsize=(8, 5))

x8  = np.arange(len(y8))
x9  = np.arange(len(y9))
x10 = np.arange(len(y10))
x11 = np.arange(len(y11))
x12 = np.arange(len(y12))

ax.plot(x8,  y8,  label='q = 0.8')
ax.plot(x9,  y9,  label='q = 0.9')
ax.plot(x10, y10, label='q = 1.0')
ax.plot(x11, y11, label='q = 1.1')
ax.plot(x12, y12, label='q = 1.2')

#ax.set_xlabel('Iteration (n)')
#ax.set_ylabel('Energy Difference')
#ax.set_title('Tsallis Energy Difference')
ax.legend(loc='lower right', shadow=True)
ax.grid(True)

ax.set_ylim(-0.0000145, 0.0000005)

plt.tight_layout()
plt.savefig('Tsallis_energy_difference.png', dpi=300)
#plt.show()
