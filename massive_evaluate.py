from config import CFG
import data_setup
import model_builder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import pandas as pd

start = time.time()
df = data_setup.get_data(data_path=CFG.DATA_PATH,
                         file_name=CFG.EVALUATE_CSV_FILE)
df = df['review']

s_bert = model_builder.load_sbert(model_path=CFG.SBERT_MODEL_PATH,
                                  sbert_path=CFG.SBERT_MODEL_FOLDER)
classifier = model_builder.get_classifier()
classifier.load_weights(f"{CFG.CLASSIFIER_MODEL_PATH}/{CFG.CLASSIFIER_MODEL_DATE}/{CFG.CLASSIFIER_MODEL_FILE}.h5")

vectorized = s_bert.encode(df.to_list())
result = classifier.predict(vectorized)

result_df = pd.concat([df, pd.DataFrame(result, columns=CFG.CLASS_NAMES)], axis=1)

result_df.to_csv(f'{CFG.DATA_PATH}/{CFG.PREDICTED_CSV}', index=False)
print(f"[INFO] Predicted csv file saved! '{CFG.DATA_PATH}/{CFG.PREDICTED_CSV}' ({time.time()-start:.3f}s)")