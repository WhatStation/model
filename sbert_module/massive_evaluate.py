from config import CFG
import data_setup
import model_builder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import pandas as pd
from tqdm.auto import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("[INFO] Loading Data and Model...")
df = data_setup.get_data(data_path=CFG.DATA_PATH,
                         file_name=CFG.PREDICT_CSV_FILE)
df = df[['name', 'review']]
reviews = df['review'].tolist()

s_bert = model_builder.load_sbert(model_path=CFG.SBERT_MODEL_PATH,
                                  sbert_path=CFG.SBERT_MODEL_FOLDER)
classifier = model_builder.SimpleMLC(n_classes=CFG.N_CLASSES)
classifier.load_state_dict(torch.load(f'{CFG.CLASSIFIER_MODEL_PATH}/{CFG.CLASSIFIER_MODEL_DATE}/{CFG.CLASSIFIER_MODEL_FILE}.pth',
                                          map_location=torch.device(device)))

print('[INFO] Done!')

print('[INFO] Evalualting...')

percentage = 0
start = time.time()
df_score = pd.DataFrame()
for i in tqdm(range(len(reviews)//128+1)):
    try:
        vectorized = s_bert.encode(reviews[i*128:(i+1)*128])#.reshape((1, -1))
        vectorized = torch.tensor(vectorized)
        result = classifier(vectorized)
        result = torch.sigmoid(result).detach().cpu().numpy()
        df_score = pd.concat([df_score, pd.DataFrame(result, columns=CFG.CLASS_NAMES)], axis=0)
    except IndexError as e:
        print(e)
        break

df = df.reset_index(drop=True)
df_score = df_score.reset_index(drop=True)

result_df = pd.concat([df, df_score], axis=1)

result_df.to_csv(f'{CFG.DATA_PATH}/{CFG.PREDICTED_CSV}', index=False)
print(f"[INFO] Predicted csv file saved! '{CFG.DATA_PATH}/{CFG.PREDICTED_CSV}' ({time.time()-start:.3f}s)")