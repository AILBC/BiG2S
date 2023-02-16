import os
import requests
import zipfile

def download(url, save_dir, file_name):
    save_path = os.path.join(save_dir, file_name)

    if not os.path.exists(save_path):
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for i in r.iter_content(chunk_size=128):
                f.write(i)

    with zipfile.ZipFile(save_path) as f:
        f.extractall(path=save_dir)