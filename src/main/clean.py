import os
import shutil

def empty_folder(folder):
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
    Lorem Ipsum
    """

    for folder in ['out/plots', 'out/models', 'data/temp']:
        empty_folder(folder)
        f = open(folder + '/.gitkeep', 'w')
        f.close()
    
if __name__ == '__main__':
    run()