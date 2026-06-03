import h5py

with h5py.File('custom_data.h5', 'r') as f:
    print(f"{'Dataset Name':<20} | {'Shape':<15} | {'Dtype':<10}")
    print("-" * 55)
    for key in f.keys():
        print(f"{key:<20} | {str(f[key].shape):<15} | {str(f[key].dtype):<10}")
