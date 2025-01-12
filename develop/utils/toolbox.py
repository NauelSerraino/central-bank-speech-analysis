import os

class ToolBox:
    
    def get_all_txt_files(self, directory):
        for root, dirs, files in os.walk(directory):
            dirs.sort(key=lambda x: int(x) if x.isdigit() else x)
            for file in files:
                if file.endswith('.txt'):
                    yield os.path.join(root, file)