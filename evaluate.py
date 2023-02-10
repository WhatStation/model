from config import CFG

import model_builder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import pandas as pd
from tabulate import tabulate


def predict_sample():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text",
                        help="write a text to predict vibe",
                        type=str)

    parser.add_argument("--model_path",
                        default=f'{CFG.CLASSIFIER_MODEL_PATH}/{CFG.CLASSIFIER_MODEL_DATE}/{CFG.CLASSIFIER_MODEL_FILE}.h5',
                        type=str)

    args = parser.parse_args()

    s_bert = model_builder.load_sbert(model_path=CFG.CLASSIFIER_MODEL_PATH,
                                    sbert_path=CFG.SBERT_MODEL_FOLDER)
    classifier = model_builder.get_classifier()
    classifier.load_weights(args.model_path)

    vectorized = s_bert.encode(args.text).reshape((1, -1))
    result = classifier.predict(vectorized)

    res = pd.DataFrame(result, columns=CFG.CLASS_NAMES).T
    res = res.rename(columns={0: 'Probability'})

    print(tabulate(res, headers='keys', tablefmt='psql'))

    tags = res[res['Probability']>0.5].index
    if len(tags) > 0:
        print(f"{', '.join(res[res['Probability']>0.5].index)} 태그를 가진 식당들을 보여줍니다")
    else:
        print("좀 더 자세한 분위기가 나타나게 작성해주세요!")


if __name__ == "__main__":
    predict_sample()