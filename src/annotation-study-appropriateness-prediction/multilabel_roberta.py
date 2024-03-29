import argparse
import math
import os
import random
import warnings
import json
import shutil
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
from spacy.lang.en import English
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers import Trainer, TrainingArguments
from transformers.models.roberta.modeling_roberta import *

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

DIMS = [
    'Appropriateness',
    'Emotions',
    'Emotional Intensity',
    'Emotional Typology',
    'Commitment',
    'Committed Seriousness',
    'Committed Openness',
    'Intelligibility',
    'Intelligible Position',
    'Intelligible Relevance',
    'Intelligible Organization',
    'Other',
    'Orthography',
    'Not classified'
]

    
class AppropriatenessDataset:
    def __init__(self, df, tokenizer, text_col, shuffle):
        self.df = df
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.shuffle = shuffle
        if self.shuffle:
            self.spacy_tokenizer = English().tokenizer

    def __getitem__(self, idx):
        argument = self.df.iloc[idx]
        text = argument[self.text_col].strip()
        if self.shuffle:
            spacy_tokenized_text = [x.text for x in self.spacy_tokenizer(text)]
            text = ' '.join(random.sample(spacy_tokenized_text, len(spacy_tokenized_text)))
        encoding = self.tokenizer(
            text,
            truncation=True,
        )

        labels = argument[DIMS].tolist()

        return {
            "id":             argument["post_id"],
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels":         labels
        }

    def __len__(self):
        return len(self.df)
    

class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if 'labels' in batch[0].keys():
            output["labels"] = [sample["labels"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
        output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
  
        # convert to tensors
        output["input_ids"]      = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if 'labels' in batch[0].keys():
            output["labels"] = torch.tensor(output["labels"], dtype=torch.float)

        return output
    

class AppropriatenessMultilabelModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weights)
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, required=False)
    parser.add_argument("--repeat", type=int, default=0, required=False)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--issue", dest='issue', action='store_true', default=False, required=False)
    parser.add_argument("--shuffle", dest='shuffle', action='store_true', default=False, required=False)
    parser.add_argument("--fix_predictions", dest='fix_predictions', action='store_true', default=False, required=False)

    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def model_init(df):
    model = AppropriatenessMultilabelModel.from_pretrained('roberta-large', num_labels=len(DIMS), problem_type="multi_label_classification")
    model.pos_weights = torch.Tensor(calc_label_weights(df)).cuda()
    return model

def calc_label_weights(df):
    pos_weights=[]
    for dim_value in df[df["fold{}.{}".format(args.repeat, args.fold)]=='TRAIN'][DIMS].values.T:
        pos_weight = (dim_value==0.).sum()/dim_value.sum()
        # handle cases where a class has only one label
        if 1 <= pos_weight < math.inf: 
            pos_weights.append(pos_weight)
        else:
            pos_weights.append(1)
    return pos_weights

def compute_metrics(fix_predictions, eval_pred):
    logits, labels = eval_pred
    probabilities = torch.sigmoid(torch.from_numpy(logits))
    predictions = torch.round(probabilities).numpy()
    probabilities = probabilities.numpy()
    if fix_predictions:
        for i, prediction in enumerate(predictions):
            tmp_prediction = np.zeros_like(prediction) 
            if prediction[DIMS.index('Appropriateness')] != 0:
                tmp_prediction[DIMS.index('Appropriateness')] = 1
                
                if prediction[DIMS.index('Emotions')] != 0:
                    tmp_prediction[DIMS.index('Emotions')] = 1
                    if prediction[DIMS.index('Emotional Intensity')] + prediction[DIMS.index('Emotional Typology')] == 0:
                       max_class = np.argmax([probabilities[i][DIMS.index('Emotional Intensity')], probabilities[i][DIMS.index('Emotional Typology')]]) 
                       if max_class == 0:
                           tmp_prediction[DIMS.index('Emotional Intensity')] = 1
                           tmp_prediction[DIMS.index('Emotional Typology')] = 0
                       else:
                           tmp_prediction[DIMS.index('Emotional Intensity')] = 0
                           tmp_prediction[DIMS.index('Emotional Typology')] = 1 
                    else:
                        tmp_prediction[DIMS.index('Emotional Intensity')] = prediction[DIMS.index('Emotional Intensity')]
                        tmp_prediction[DIMS.index('Emotional Typology')] = prediction[DIMS.index('Emotional Typology')]
                
                if prediction[DIMS.index('Commitment')] != 0:
                    tmp_prediction[DIMS.index('Commitment')] = 1
                    if prediction[DIMS.index('Committed Seriousness')] + prediction[DIMS.index('Committed Openness')] == 0:
                       max_class = np.argmax([probabilities[i][DIMS.index('Committed Seriousness')], probabilities[i][DIMS.index('Committed Openness')]]) 
                       if max_class == 0:
                           tmp_prediction[DIMS.index('Committed Seriousness')] = 1
                           tmp_prediction[DIMS.index('Committed Openness')] = 0
                       else:
                           tmp_prediction[DIMS.index('Committed Seriousness')] = 0
                           tmp_prediction[DIMS.index('Committed Openness')] = 1 
                    else:
                        tmp_prediction[DIMS.index('Committed Seriousness')] = prediction[DIMS.index('Committed Seriousness')]
                        tmp_prediction[DIMS.index('Committed Openness')] = prediction[DIMS.index('Committed Openness')]

                if prediction[DIMS.index('Intelligibility')] != 0:
                    tmp_prediction[DIMS.index('Intelligibility')] = 1
                    if prediction[DIMS.index('Intelligible Position')] + prediction[DIMS.index('Intelligible Relevance')] + prediction[DIMS.index('Intelligible Organization')] == 0:
                       max_class = np.argmax([probabilities[i][DIMS.index('Intelligible Position')], probabilities[i][DIMS.index('Intelligible Relevance')], prediction[DIMS.index('Intelligible Organization')]]) 
                       if max_class == 0:
                           tmp_prediction[DIMS.index('Intelligible Position')] = 1
                           tmp_prediction[DIMS.index('Intelligible Relevance')] = 0
                           tmp_prediction[DIMS.index('Intelligible Organization')] = 0
                       elif max_class == 1:
                           tmp_prediction[DIMS.index('Intelligible Position')] = 0
                           tmp_prediction[DIMS.index('Intelligible Relevance')] = 1 
                           tmp_prediction[DIMS.index('Intelligible Organization')] = 0
                       else:
                           tmp_prediction[DIMS.index('Intelligible Position')] = 0
                           tmp_prediction[DIMS.index('Intelligible Relevance')] = 0 
                           tmp_prediction[DIMS.index('Intelligible Organization')] = 1
                    else:
                        tmp_prediction[DIMS.index('Intelligible Position')] = prediction[DIMS.index('Intelligible Position')]
                        tmp_prediction[DIMS.index('Intelligible Relevance')] = prediction[DIMS.index('Intelligible Relevance')]
                        tmp_prediction[DIMS.index('Intelligible Organization')] = prediction[DIMS.index('Intelligible Organization')]

                if prediction[DIMS.index('Other')] != 0:
                    tmp_prediction[DIMS.index('Other')] = 1
                    if prediction[DIMS.index('Orthography')] + prediction[DIMS.index('Not classified')] == 0:
                       max_class = np.argmax([probabilities[i][DIMS.index('Orthography')], probabilities[i][DIMS.index('Not classified')]]) 
                       if max_class == 0:
                           tmp_prediction[DIMS.index('Orthography')] = 1
                           tmp_prediction[DIMS.index('Not classified')] = 0
                       else:
                           tmp_prediction[DIMS.index('Orthography')] = 0
                           tmp_prediction[DIMS.index('Not classified')] = 1 
                    else:
                        tmp_prediction[DIMS.index('Orthography')] = prediction[DIMS.index('Orthography')]
                        tmp_prediction[DIMS.index('Not classified')] = prediction[DIMS.index('Not classified')]
            predictions[i] = tmp_prediction

    out_dict = {}
    prec = 0
    rec = 0
    macroF1 = 0
    for i, dim in enumerate(DIMS):
        scores = precision_recall_fscore_support(labels[:,i], predictions[:,i], average='macro')
        prec += scores[0]
        rec += scores[1]
        macroF1 += scores[2]
        out_dict[dim+'_precision'] = scores[0]
        out_dict[dim+'_recall'] = scores[1]
        out_dict[dim+'_macroF1'] = scores[2]
        
    out_dict['mean_precision'] = prec/len(DIMS)
    out_dict['mean_recall'] = rec/len(DIMS)
    out_dict['mean_F1'] = macroF1/len(DIMS)
            
    return out_dict

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.input)
    
    if args.issue:
        text_col = 'arg_issue'
    else:
        text_col = 'post_text'      
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    train_dataset = AppropriatenessDataset(df[df["fold{}.{}".format(args.repeat, args.fold)]=='TRAIN'], tokenizer, text_col, args.shuffle)
    valid_dataset = AppropriatenessDataset(df[df["fold{}.{}".format(args.repeat, args.fold)]=='VALID'], tokenizer, text_col, args.shuffle)
    test_dataset  = AppropriatenessDataset(df[df["fold{}.{}".format(args.repeat, args.fold)]=='TEST'], tokenizer, text_col, args.shuffle)
    
    training_args = TrainingArguments(
    output_dir=os.path.join(args.output, f"fold{args.repeat}.{args.fold}"),
    learning_rate=3e-6,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=250,
    save_steps=250,
    load_best_model_at_end=True,
    num_train_epochs=20,
    max_steps=-1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.1,
    logging_dir=os.path.join(args.output, f"fold{args.repeat}.{args.fold}")+'/logs',
    skip_memory_metrics=True,
    disable_tqdm=False,
    metric_for_best_model='mean_F1',
    greater_is_better=True,
    warmup_ratio=0.1,
    lr_scheduler_type="polynomial",
    )

    trainer = Trainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=Collate(tokenizer=tokenizer),
        tokenizer=tokenizer,
        model_init=partial(model_init, df),
        compute_metrics=partial(compute_metrics, args.fix_predictions)
    )
    
    trainer.train()
    trainer.eval_dataset = test_dataset
    metrics = trainer.evaluate()
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

