from config import CFG
from data_setup import dataloader_for_sbert
from model_builder import load_sbert

import math
import datetime
import torch
from sentence_transformers import losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator

def sbert_train(train_csv_file=CFG.MODEL_CSV_FILE): 
    now = datetime.datetime.now().strftime("%y-%m-%d_%H%M")

    # Load model
    s_bert = load_sbert(model_path=CFG.SBERT_MODEL_PATH,
                        sbert_path=CFG.SBERT_MODEL_FOLDER)

    # Set Dataloader
    sbert_train_dataloader = dataloader_for_sbert(CFG.DATA_PATH, train_csv_file,
                                            batch_size=CFG.BERT_BATCH_SIZE,
                                            test_size=0.2)
    # sbert_evaluator = BinaryClassificationEvaluator.from_input_examples(
    #     dataloader_for_sbert(
    #         CFG.DATA_PATH, CFG.MODEL_CSV_FILE,
    #         batch_size=CFG.BERT_BATCH_SIZE,
    #         test_size=0.2,
    #         is_train=False
    #         ),
    #         name='sbert_eval'
    #     )

    # print(next(iter(sbert_dataloader)))

    # Loss function
    train_loss = losses.CosineSimilarityLoss(s_bert)

    # warming up learning rate during 10% of training progress
    warmup_step = math.ceil(len(sbert_train_dataloader) * CFG.BERT_EPOCHS * 0.1)

    # Train SBERT
    s_bert.fit(train_objectives=[(sbert_train_dataloader, train_loss)],
            epochs=CFG.BERT_EPOCHS,
            warmup_steps=warmup_step,
            #    evaluator=sbert_evaluator,
            #    evaluation_steps=int(len(sbert_train_dataloader)*0.1),
            output_path=f"{CFG.SBERT_MODEL_PATH}/{now}",
            #    optimizer_class=torch.optim.AdamW,
            #    optimizer_params={'lr': 2e-5},
            #    save_best_model=True,
            #    weight_decay=0.01
            )
    
    return now

if __name__ == "__main__":
    sbert_train()