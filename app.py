import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

def load_and_prepare_text():
    """Загрузка и подготовка текста 'Евгения Онегина'"""
    onegin_text = """
    Мой дядя самых честных правил,
    Когда не в шутку занемог,
    Он уважать себя заставил
    И лучше выдумать не мог.
    Его пример другим наука;
    Но, боже мой, какая скука
    С больным сидеть и день и ночь,
    Не отходя ни шагу прочь!
    Какое низкое коварство
    Полуживого забавлять,
    Ему подушки поправлять,
    Печально подносить лекарство,
    Вздыхать и думать про себя:
    Когда же черт возьмет тебя!
    
    Так думал молодой повеса,
    Летя в пыли на почтовых,
    Всевышней волею Зевеса
    Наследник всех своих родных.
    Друзья Людмилы и Руслана!
    С героем моего романа
    Без предисловий, сей же час
    Позвольте познакомить вас:
    Онегин, добрый мой приятель,
    Родился на брегах Невы,
    Где, может быть, родились вы
    Или блистали, мой читатель;
    Там некогда гулял и я:
    Но вреден север для меня.
    
    Я к вам пишу — чего же боле?
    Что я могу еще сказать?
    Теперь, я знаю, в вашей воле
    Меня презреньем наказать.
    """
    
    # Разделяем текст на примеры
    samples = [stanza.strip() for stanza in onegin_text.split('\n\n') if stanza.strip()]
    return samples

def tokenize_function(examples):
    """Функция токенизации для датасета"""
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding=True, 
        max_length=256,
        return_special_tokens_mask=True
    )

def main():
    global tokenizer
    
    print("Загружаем модель и токенизатор...")
    
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("Подготавливаем данные...")
    texts = load_and_prepare_text()
    
    # Создаём датасет
    dataset = Dataset.from_dict({"text": texts})
    
    # Токенизируем данные
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Исправленные параметры обучения (совместимые с новыми версиями)
    training_args = TrainingArguments(
        output_dir="./onegin_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        # Исправленные параметры для новых версий:
        eval_strategy="no",  # вместо evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=False,
        prediction_loss_only=True,
        report_to=None,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Создаём тренер
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Начинаем обучение...")
    trainer.train()
    
    # Сохраняем модель
    # 
    
    trainer.save_model()
    tokenizer.save_pretrained("./onegin_model")
    print("Модель сохранена в папке './onegin_model'")
    print("Модель сохранена в папке './onegin_model'")
    
    # Тестируем модель
    test_generation()

def test_generation():
    """Функция для тестирования обученной модели"""
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*50)
    
    # Загружаем обученную модель
    model = AutoModelForCausalLM.from_pretrained("./onegin_model")
    tokenizer = AutoTokenizer.from_pretrained("./onegin_model")
    
    model.eval()
    
    test_prompts = [
        "Мой дядя самых честных правил,",
        "Онегин, добрый мой приятель,",
        "Я к вам пишу - чего же боле?",
    ]
    
    for prompt in test_prompts:
        print(f"\nПромпт: '{prompt}'")
        print("Генерация:")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=100,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        print("-" * 80)

if __name__ == "__main__":
    main()