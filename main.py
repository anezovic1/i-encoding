import numpy as np

def IEncoder(n_categories):
    
    theta = 360 / n_categories
    theta_arr = np.arange(0, 360, theta)

    z_arr = np.exp(1j * np.deg2rad(theta_arr))
    
    # QUESTION: what do steps 3 and 4 mean?
    # phases = np.angle(z_arr, deg=True)

    phases = theta_arr

    f_encodings = np.cos(np.deg2rad(phases))

    theta_arr = np.round(theta_arr, 2)
    z_arr = np.round(z_arr, 2)
    phases = np.round(phases, 2)
    f_encodings = np.round(f_encodings, 2)

    return theta_arr, z_arr, phases, f_encodings


n_categories = 8
theta_arr, z_arr, phases, f_encodings = IEncoder(n_categories)

print("theta array (degrees):", theta_arr)

print("[")
for c in z_arr:
    print(c)
print("]")

print("phases (degrees):", phases)
print("feature encodings:", f_encodings)
