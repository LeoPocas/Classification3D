from Classification3D.utils import OUTPUT_PATH
import os

for filename in os.listdir(OUTPUT_PATH):
    if filename.endswith('.png') or filename.endswith('.gif'):
        file = os.path.join(OUTPUT_PATH, filename)
        os.remove(file)