import math
import datetime

from torch.utils.data import DataLoader

from sentence_transformers import losses
from sentence_transformers.readers import InputExample

from config import CFG
import data_setup
import model_builder

def sbert_train():
    num_epochs = CFG.BERT_EPOCHS
    now = datetime.datetime.now().strftime("%y-%m-%d_%H%M")

    # Prepare data
    df = data_setup.get_data(CFG.DATA_PATH, CFG.SBERT_CSV_FILE)
    X_train, _, _, _ = data_setup.data_split(
        df,
        random_state=CFG.RANDOM_STATE
    )
    df_melt = data_setup.long_form(df, X_train[:, 0])

    # Array for inputs
    gold_samples = []

    # Create Input for SBERT
    for i in range(len(df_melt)):
        gold_samples.append(InputExample(texts=[df_melt.loc[i, 'review'],
                                                df_melt.loc[i, 'variable']],
                                        label=float(df_melt.loc[i, 'value'])))

    # Load SBERT
    s_bert = model_builder.load_sbert(model_path=CFG.SBERT_MODEL_PATH, 
                                      sbert_path=CFG.SBERT_MODEL_FOLDER)

    # Set DataLoader and Loss function
    train_dataloader = DataLoader(gold_samples, 
                                  shuffle=True,
                                  batch_size=CFG.BERT_BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(s_bert)

    # configure the training
    # warming up learning rate during 10% of training progress
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

    # Train SBERT
    s_bert.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=f"{CFG.SBERT_MODEL_PATH}/{now}")


if __name__ == "__main__":
    sbert_train()