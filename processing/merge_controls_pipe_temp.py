import pandas as pd
from scipy.io import loadmat


if __name__ == "__main__":
    # Load the CSV file
    csv_file_path = 'data/AgriControl/comparison/gl_u_data.csv'
    csv_data = pd.read_csv(csv_file_path)

    # Select specific columns from the CSV
    csv_selected_columns = csv_data[['boil', 'extCo2', 'thScr', 'roof', 'lamp', 'blScr']]

    # Load the MAT file
    mat_file_path = 'data/AgriControl/comparison/gl_controls.mat'
    mat_data = loadmat(mat_file_path)

    # Extract the 'controls' variable
    controls_data = mat_data['controls']

    # Extract the fifth column (index 4) representing pipe temperature
    pipe_temperature = controls_data[:, 4]

    # Convert to DataFrame for easy merging
    pipe_temperature_df = pd.DataFrame(pipe_temperature, columns=['pipe_temperature'])

    # Combine both DataFrames
    merged_data = pd.concat([csv_selected_columns, pipe_temperature_df], axis=1)

    # Save the merged data to a new CSV file
    output_file_path = 'merged_data.csv'
    merged_data.to_csv(output_file_path, index=False)
