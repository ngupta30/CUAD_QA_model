import pandas as pd
import datasets as ds
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
import torch
from predict import run_prediction
from transformers import default_data_collator


def get_answer_from_large_context(ques, context_file, model_path, output_file):
    # read the context file
    with open(context_file) as f:
        lines = f.readlines()
    context = ' '.join(lines)
    # prediction
    predictions = run_prediction(ques, context, model_path)
    with open(output_file, 'w') as f:
        for i, p in enumerate(predictions):
            f.write(f"Question {i+1}: {ques[int(p)]}\nAnswer: {predictions[p]}\n\n")
    return predictions


def get_answer_from_small_context(ques, context_file, model_path, output_file):
    # read the context file
    with open(context_file) as f:
        lines = f.readlines()
    context = ' '.join(lines)
    # model & Tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # prediction
    answer_list = []
    for q in ques:
        encoding = tokenizer.encode_plus(text=q, text_pair=context)
        inputs = encoding['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(inputs)
        outputs = model(input_ids=torch.tensor([inputs]))
        # output
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1]).strip()
        answer_list.append(answer)
    
    with open(output_file, 'w') as f:
        for i in range(len(ques)):
            f.write(f"Question {i+1}: {ques[i]}\nAnswer: {answer_list[i]}\n\n")
    return answer_list


# this is a sample dataset format required for training
'''
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'}
 '''
# to convert CUAD data in required format for training

def prepare_training_data(data):
    questions = []
    answers_list =[]
    for i, q in enumerate(data['qas']):
        ans = data['qas'][i]['answers']
        question = data['qas'][i]['question']
        dd ={'text':[], 'answer_start':[]}
        for i in ans:
            dd['text'].append(i['text'])
            dd['answer_start'].append(i['answer_start'])
        questions.append(question)
        answers_list.append(dd)
    contract = data['context']
    train_data = pd.DataFrame(list(zip(questions, answers_list)))
    train_data.columns = ['question', 'answer']
    train_data['context'] = contract
    return train_data.to_dict('r')

def split_train_valid_dataset(data_dict, n_train):
    df = pd.DataFrame(data_dict)
    df_train = df[:n_train].reset_index(drop=True)
    df_valid = df[n_train:].reset_index(drop=True)
    train_dataset = ds.Dataset.from_pandas(df_train)
    valid_dataset = ds.Dataset.from_pandas(df_valid)
    main_data = ds.DatasetDict({"train":train_dataset,"test":valid_dataset})
    return main_data


def prepare_train_features(examples,model_path, max_length = 512, stride = 100):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pad_on_right = tokenizer.padding_side == "right"
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answer"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def fine_tune_model(data_dict, n_train, max_length, stride, model_path,new_model_name,
                    batch_size = 16, learning_rate=2e-5):
    #df_dict = prepare_training_data(data_dict)
    main_data = split_train_valid_dataset(data_dict, n_train)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_features = prepare_train_features(main_data['train'][:], model_path,max_length, stride)
    train_dict= ds.Dataset.from_dict(train_features)
    vaild_features = prepare_train_features(main_data['test'][:], model_path,max_length, stride)
    vaild_dict= ds.Dataset.from_dict(vaild_features)

    args = TrainingArguments(
    new_model_name+'_logs',
    evaluation_strategy = "epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    #push_to_hub=True,
    )
    data_collator = default_data_collator
    trainer = Trainer(
        model,
        args,
        train_dataset= train_dict, 
        eval_dataset= vaild_dict,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(new_model_name)


