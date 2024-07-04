import os
import pickle
with open(os.path.join("/home/public/d4j/feature/Chart/1b/static_fea"), 'rb') as f:
    staticFea = pickle.load(f)
print(staticFea)