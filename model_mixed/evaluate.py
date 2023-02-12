from config import CFG

import model_builder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pandas as pd
from tabulate import tabulate
tabulate.WIDE_CHARS_MODE = False

def predict_sample():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--text",
                        help="write a text to predict vibe",
                        type=str)

    parser.add_argument("--model_path",
                        default=f'{CFG.CLASSIFIER_MODEL_PATH}/{CFG.CLASSIFIER_MODEL_DATE}/{CFG.CLASSIFIER_MODEL_FILE}.h5',
                        type=str)

    args = parser.parse_args()

    tagged_store = pd.read_csv(f"{CFG.DATA_PATH}/{CFG.TAG_CSV}")

    s_bert = model_builder.load_sbert(model_path=CFG.SBERT_MODEL_PATH,
                                      sbert_path=CFG.SBERT_MODEL_FOLDER)
    classifier = model_builder.get_classifier()
    classifier.load_weights(args.model_path)

    vectorized = s_bert.encode(args.text).reshape((1, -1))
    result = classifier.predict(vectorized)

    res = pd.DataFrame(result, columns=CFG.CLASS_NAMES).T
    res = res.rename(columns={0: 'Probability'})

    convert_col = dict(zip(tagged_store.iloc[:, -16:-1].columns, CFG.CLASS_NAMES))
    tagged_store = tagged_store.rename(columns=convert_col)

    store_tag_prob = pd.concat([tagged_store['name'], tagged_store.iloc[:, -16:-1] * res['Probability']], axis=1)
    store_tag_prob['sum'] = store_tag_prob.iloc[:, -16:-1].sum(axis=1, numeric_only=True)


    tags = res[res['Probability']>0.5].index
    if len(tags) > 0:
        print(f"{', '.join(res[res['Probability']>0.5].index)} 태그를 가진 식당들을 보여줍니다")
    else:
        print("좀 더 자세한 분위기가 나타나게 작성해주세요!")


    if store_tag_prob['sum'].max() < 1:
        print("좀 더 자세한 분위기가 나타나게 작성해주세요!")
    else:
        store_tag_prob = store_tag_prob.sort_values('sum', ascending=False)
        store_tag_prob = store_tag_prob.T.iloc[:, :5]

        store_tag_prob.columns = store_tag_prob.iloc[0]
        store_tag_prob = store_tag_prob.drop(store_tag_prob.index[0])
        print(tabulate(store_tag_prob, headers='keys', tablefmt='psql', floatfmt=".4f"))

if __name__ == "__main__":
    predict_sample()
