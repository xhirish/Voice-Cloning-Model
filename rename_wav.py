import os



def rename_wave_files(folder_path):
    files = os.listdir(folder_path)
    wav_files = [f for f in files if f.lower().endswith('.wav')]

    print(f'Folder path: {folder_path}')
    print(f'All files in the folder: {files}')
    print(f'Wave files found: {len(wav_files)}')

    for index, wav_file in enumerate(wav_files, start=1):
        old_path = os.path.join(folder_path, wav_file)
        new_filename = f'{index}.wav'
        new_path = os.path.join(folder_path, get_new_filename(folder_path, new_filename))
        
        os.rename(old_path, new_path)
        print(f'Renamed {old_path} to {new_path}')

def get_new_filename(folder_path, filename):
    new_filename = filename
    counter = 1

    while os.path.exists(os.path.join(folder_path, new_filename)):
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{counter}{ext}"
        counter += 1

    return new_filename

if __name__ == "__main__":
    folder_path = r'D:\\Machine Learning Projects\\OpeninAPP22\wavs'
    rename_wave_files(folder_path)
