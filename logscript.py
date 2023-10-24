import os
import json
import csv

def get_data(directory):
    command_file = os.path.join(directory, 'command')
    result_file = os.path.join(directory, 'result')

    if not (os.path.exists(command_file) and os.path.exists(result_file)):
        return None

    with open(command_file, 'r') as cmd_file:
        command_data = json.load(cmd_file)

    with open(result_file, 'r') as res_file:
        result_data = res_file.read()
        json_start = result_data.find('{')
        json_end = result_data.rfind('}') + 1
        result_data = result_data[json_start:json_end]
        result_data = json.loads(result_data)

    combined_data = {**command_data, **result_data}
    return combined_data

def main(root_dir, output_file):
    data_list = []
    for subdir, dirs, files in os.walk(root_dir):
        if 'command' in files and 'result' in files:
            combined_data = get_data(subdir)
            if combined_data:
                data_list.append(combined_data)

    if not data_list:
        print("No data found.")
        return

    # Dynamically generate fieldnames based on keys in the dictionaries
    keys = set().union(*(d.keys() for d in data_list))
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_list)
    print(f"CSV generated successfully at {output_file}")



if __name__ == "__main__":
    root_dir = "./saved_exp"  # Replace with the root directory path
    output_file = "logs.csv"  # Output file name
    main(root_dir, output_file)
