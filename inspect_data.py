import numpy as np

# Load your file
data = np.load('C:\\Users\\GIGABYTE\\PPP-drone\\dataset\\train\\had_collision.npy')

# Print the essential metadata
print(f"Shape of data: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"First 5 rows:\n{data[:5]}")

# If you want to see the min/max to check the scale
print(f"Min value: {np.min(data)}")
print(f"Max value: {np.max(data)}")