import os

#Goal: Extract satellite name and frame number from filenames and sort them into separate lists
#Filename are the output of pyGCS_raytrace_eeggl.py
#Filenames are synthetic images generate by raytrace using pyGCS parameters based ond pyGCS cloud model.
#dir is the directory containing the files
#sat is the satellite number (1, 2, or 3): 1 for COR2A, 2 for COR2B, 3 for SECCHI
#Output: 3 lists of filenames, one for each satellite, sorted by frame number

def btot_file_sorted(dir,instr):
    # Initialize lists for each satellite
    sat1_files = []
    sat2_files = []
    sat3_files = []

    if instr == 'cor2_a':
        sat = 1
    elif instr == 'cor2_b':
        sat = 2
    elif instr == 'lascoC2':
        sat = 3

    # Iterate through files in the folder
    for filename in os.listdir(dir):
        # Extract satellite number from the filename
        if filename.endswith('.fits'):
            satellite_number = filename.split('_')[6][-1]

            # Extract frame number from the filename
            frame_number = filename.split('_')[8][-1]

            # Append the filename to the corresponding satellite list
            if satellite_number == '1':
                sat1_files.append((frame_number, filename))
            elif satellite_number == '2':
                sat2_files.append((frame_number, filename))
            elif satellite_number == '3':
                sat3_files.append((frame_number, filename))

    # Sort files in each satellite list by frame number
    sat1_files.sort()
    sat2_files.sort()
    sat3_files.sort()

    sat1_sorted = [file[1] for file in sat1_files]
    sat2_sorted = [file[1] for file in sat2_files]
    sat3_sorted = [file[1] for file in sat3_files]
    
    if sat == 1:
        return sat1_sorted
    elif sat == 2:  
        return sat2_sorted
    elif sat == 3:
        return sat3_sorted