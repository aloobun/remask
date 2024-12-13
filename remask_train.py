import torch
import random
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

class ReMaskTrainer:
    def __init__(self, model_name="amd/AMD-Llama-135m", learning_rate=1e-5):
        self.prefix = [
            "So, the answer is ",
            "Therefore, the answer is ",
            "As a result, the answer is ",
            "On this basis, ",
            "Considering this, we get ",
            "Putting this all together, we get: ",
            "After all that, it seems that the answer is "
        ]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        special_tokens_dict = {
            "additional_special_tokens": ["<|user|>", "<|logic|>", "<|answer|>"]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.tokenizer.mask_token = self.tokenizer.eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2
        )
    
    def compute_remask_loss(self, input_ids, labels, logic_mask):
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        logic_mask = logic_mask.to(self.device)
        
        eps = 1e-8
        
        masked_input_ids = input_ids.clone()
        mask_token_id = self.tokenizer.mask_token_id
        masked_input_ids[logic_mask] = mask_token_id
        
        full_output = self.model(input_ids, labels=labels)
        masked_output = self.model(masked_input_ids, labels=labels)
        
        p_full = torch.clamp(F.softmax(full_output.logits, dim=-1), min=eps, max=1-eps)
        p_masked = torch.clamp(F.softmax(masked_output.logits, dim=-1), min=eps, max=1-eps)
        
        kl_forward = F.kl_div(
            torch.log(p_masked), p_full, reduction='batchmean'
        )
        kl_backward = F.kl_div(
            torch.log(p_full), p_masked, reduction='batchmean'
        )
        divergence_loss = 0.5 * (kl_forward + kl_backward)
        
        ce_loss = 0.5 * (full_output.loss + masked_output.loss)
        total_loss = ce_loss + 0.1 * divergence_loss
        
        return total_loss
    
    def preprocess_data(self, example):
        try:
            instruction = example.get('prompt', '')
            reasoning = example.get('rationale', '')
            answer = example.get('target', '')
            
            if not (instruction and answer):
                return None
            prfx = random.choice(self.prefix)
            augmented_answer = prfx + answer
            
            prompt = f"<|user|>\n{instruction}\n<|logic|>\n{reasoning}\n<|answer|>\n{augmented_answer}"
            return {"text": prompt}
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def train(self, dataset_name="euclaise/TinyCoT", epochs=6, batch_size=4):
        dataset = load_dataset(dataset_name)
        formatted_dataset = dataset['train'].map(self.preprocess_data).filter(lambda x: x is not None)
        
        train_loader = DataLoader(formatted_dataset, batch_size=batch_size, shuffle=True)
        
        if len(train_loader) == 0:
            raise ValueError("No valid training data found. Check your dataset preprocessing.")
        
        self.model.train()
        global_step = 0
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                try:
                    inputs = self.tokenizer(
                        batch['text'], 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512
                    )
                    
                    labels = inputs['input_ids'].clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    
                    logic_start = inputs['input_ids'] == self.tokenizer.encode("<|logic|>", add_special_tokens=False)[0]
                    logic_end = inputs['input_ids'] == self.tokenizer.encode("<|answer|>", add_special_tokens=False)[0]
                    logic_mask = (logic_start.cumsum(dim=1) > 0) & (logic_end.cumsum(dim=1) == 0)
                    
                    loss = self.compute_remask_loss(inputs['input_ids'], labels, logic_mask)
                    
                    if torch.isnan(loss):
                        print("NaN loss detected, skipping batch")
                        continue
                    
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    total_loss += loss.item()
                    if global_step % 100 == 0:
                        print(f"Step {global_step}, Loss: {loss.item()}")
                
                except Exception as e:
                    print(f"Error in training step: {e}")
                    continue
            
            self.scheduler.step(total_loss / len(train_loader))
            
            print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}")
        
        self.model.save_pretrained("remask_trained_model")
        self.tokenizer.save_pretrained("remask_trained_model")

if __name__ == "__main__":
    trainer = ReMaskTrainer()
    trainer.train()
