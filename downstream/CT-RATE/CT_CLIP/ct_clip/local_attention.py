import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from ct_clip.sentence_pool import SentenceAttentionPool


class MLMCrossMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, img_size=96, drop_rate=0):
        super(MLMCrossMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query1 = nn.Linear(embed_dim, embed_dim)
        self.key1 = nn.Linear(embed_dim, embed_dim)
        self.value1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.query2 = nn.Linear(embed_dim, embed_dim)
        self.key2 = nn.Linear(embed_dim, embed_dim)
        self.value2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(drop_rate)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(250, 1)
        if img_size == 96:
            self.fc2 = nn.Linear(27, 1)
        elif img_size == 128:
            self.fc2 = nn.Linear(48, 1)
        elif img_size == 160:
            self.fc2 = nn.Linear(50, 1)
        elif img_size == 192:
            self.fc2 = nn.Linear(36, 1)

    def forward(
        self,
        input_tensor1,
        input_tensor2,
        attention_mask1=None,
        attention_mask2=None      
    ):
        # for vision input [B, I]
        query_layer1 = self.query1(input_tensor1)
        key_layer1 = self.key1(input_tensor1)
        value_layer1 = self.value1(input_tensor1)


        # for text input [B, T]
        query_layer2 = self.query2(input_tensor2)
        key_layer2 = self.key2(input_tensor2)
        value_layer2 =  self.value2(input_tensor2)

        
        attention_scores1  = query_layer2 @ key_layer1.T # [T, D] @ [D, I] = [T, I]
        attention_scores1 = attention_scores1 / math.sqrt(self.embed_dim)
        if attention_mask1 is not None:
            attention_scores1 = attention_scores1 + attention_mask1

        # Sigmoid is better in this case
        # TODO: pre-normalize vs. post-normalize 
        attention_probs1 = F.sigmoid(attention_scores1)
    
        attention_probs1 = self.dropout1(attention_probs1)
        
        context_layer1_whole, context_layer1 = [], []
        for i in range(attention_probs1.size(1)):
            context_layer1_whole.append((attention_probs1[:, i].unsqueeze(1) * value_layer2).unsqueeze(0))
        context_layer1_whole = torch.cat(context_layer1_whole, dim=0).permute(1, 0, 2)
        
        for t in range(attention_probs1.size(0)):
            context_layer1.append(self.fc1(context_layer1_whole[t, :, :].transpose(0,1)).transpose(0,1))
        context_layer1 = torch.cat(context_layer1, dim=0)
        
        attention_scores2 = query_layer1 @ key_layer2.T # [I, D] @ [D, T] = [I, T]
        attention_scores2 = attention_scores2 / math.sqrt(self.embed_dim)

        if attention_mask2 is not None:
            attention_scores2 = attention_scores2 + attention_mask2
       
        attention_probs2 = F.sigmoid(attention_scores2)

        attention_probs2 = self.dropout2(attention_probs2)
        
        context_layer2 = attention_probs2 @ value_layer2 # [I, T] @ [T, D] = [I, D]
        context_layer2_whole, context_layer2 = [], []
        
        for t in range(attention_probs2.size(1)):
            context_layer2_whole.append((attention_probs2[:, t].unsqueeze(1) * value_layer1).unsqueeze(0))
        context_layer2_whole = torch.cat(context_layer2_whole, dim=0).permute(1, 0, 2)
        
        for i in range(attention_probs2.size(0)):
            context_layer2.append(self.fc2(context_layer2_whole[i, :, :].transpose(0,1)).transpose(0,1))
        context_layer2 = torch.cat(context_layer2, dim=0)
        
        return context_layer2, attention_probs2, context_layer1, attention_probs1


class MIMCrossMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, img_size=96, drop_rate=0):
        super(MIMCrossMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query1 = nn.Linear(embed_dim, embed_dim)
        self.key1 = nn.Linear(embed_dim, embed_dim)
        self.value1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.query2 = nn.Linear(embed_dim, embed_dim)
        self.key2 = nn.Linear(embed_dim, embed_dim)
        self.value2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(drop_rate)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if img_size == 96:
            self.fc1 = nn.Linear(27, 1)
        elif img_size == 128:
            self.fc1 = nn.Linear(48, 1)
        elif img_size == 160:
            self.fc1 = nn.Linear(50, 1)
        elif img_size == 192:
            self.fc1 = nn.Linear(36, 1)
            
        self.fc2 = SentenceAttentionPool(32, 768, pos_embed=False) # Max sentence num: 32

    def forward(
        self,
        input_tensor1,
        input_tensor2,
        attention_mask1=None,
        attention_mask2=None      
    ):
        # for vision input [B, I]
        query_layer1 = self.query1(input_tensor1)
        key_layer1 = self.key1(input_tensor1)
        value_layer1 = self.value1(input_tensor1)


        # for text input [B, T]
        query_layer2 = self.query2(input_tensor2)
        key_layer2 = self.key2(input_tensor2)
        value_layer2 =  self.value2(input_tensor2)

        
        attention_scores1  = query_layer2 @ key_layer1.T # [T, D] @ [D, I] = [T, I]
        attention_scores1 = attention_scores1 / math.sqrt(self.embed_dim)
        if attention_mask1 is not None:
            attention_scores1 = attention_scores1 + attention_mask1

        # Sigmoid is better in this case
        # TODO: pre-normalize vs. post-normalize 
        attention_probs1 = F.sigmoid(attention_scores1)
    
        attention_probs1 = self.dropout1(attention_probs1)
        
        context_layer1_whole, context_layer1 = [], []
        for i in range(attention_probs1.size(1)):
            context_layer1_whole.append((attention_probs1[:, i].unsqueeze(1) * value_layer2).unsqueeze(0))
        context_layer1_whole = torch.cat(context_layer1_whole, dim=0).permute(1, 0, 2)
        
        for t in range(attention_probs1.size(0)):
            context_layer1.append(self.avgpool(context_layer1_whole[t, :, :].transpose(0,1)).transpose(0,1))
        context_layer1 = torch.cat(context_layer1, dim=0)
        
        attention_scores2 = query_layer1 @ key_layer2.T # [I, D] @ [D, T] = [I, T]
        attention_scores2 = attention_scores2 / math.sqrt(self.embed_dim)

        if attention_mask2 is not None:
            attention_scores2 = attention_scores2 + attention_mask2
       
        attention_probs2 = F.sigmoid(attention_scores2)

        attention_probs2 = self.dropout2(attention_probs2)
        
        context_layer2 = attention_probs2 @ value_layer2 # [I, T] @ [T, D] = [I, D]
        context_layer2_whole, context_layer2 = [], []
        
        for t in range(attention_probs2.size(1)):
            context_layer2_whole.append((attention_probs2[:, t].unsqueeze(1) * value_layer1).unsqueeze(0))
        context_layer2_whole = torch.cat(context_layer2_whole, dim=0).permute(1, 0, 2)
        
        for i in range(attention_probs2.size(0)):
            context_layer2.append(self.fc2(context_layer2_whole[i, :, :].unsqueeze(0)))
        context_layer2 = torch.cat(context_layer2, dim=0)
        
        return context_layer2, attention_probs2, context_layer1, attention_probs1