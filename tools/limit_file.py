import os

def truncate_file_content(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r+') as file:
                content = file.read()
                truncated_content = ' '.join(content.split(" ")[:1000])
                file.seek(0)
                file.truncate(0)
                file.write(truncated_content)

truncate_file_content('datasets/bsnc')
truncate_file_content('datasets/bsmc')
truncate_file_content('datasets/bsc')
truncate_file_content('datasets/ssrl')
truncate_file_content('datasets/bep')
