import os
import shutil

def empty_folder(folder):
    """
    Removes all files in the specified directory including other directories

    Parameters
    ----------
    folder : str
        Path to the folder to be emptied.
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def run():
    """
    Main function that iterates over predefined folders, empties them, and creates .gitkeep file.
    """
    for folder in ['src/data/temp', 'out/plots', 'out/models', 'out/forecast_tables']:
        empty_folder(folder)
        f = open(folder + '/.gitkeep', 'w')
        f.close()
    
if __name__ == '__main__':
    run()