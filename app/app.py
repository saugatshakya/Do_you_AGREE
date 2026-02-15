from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import json
import re
import os
import math

app = Flask(__name__)

# Load configurations and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
with open('../models/vocab.json', 'r') as f:
    vocab_config = json.load(f)
vocab_size = vocab_config['vocab_size']
max_len = vocab_config['max_len']

with open('../models/word2id.json', 'r') as f:
    word2id = json.load(f)

# Define model classes (same as in notebook)
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device
    def forward(self,x,seg):
        x = torch.clamp(x, 0, self.tok_embed.num_embeddings - 1)
        pos = torch.arange(x.size(1), device=self.device).unsqueeze(0).expand_as(x)
        return self.norm(self.tok_embed(x)+self.pos_embed(pos)+self.seg_embed(seg))

def get_attn_pad_mask(q,k,device):
    return k.eq(0).unsqueeze(1).to(device).expand(q.size(0),q.size(1),k.size(1))

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k,device):
        super().__init__(); self.scale = math.sqrt(d_k); self.device=device
    def forward(self,Q,K,V,mask):
        scores = torch.matmul(Q,K.transpose(-1,-2))/self.scale
        scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads,d_model,d_k,device):
        super().__init__()
        self.n_heads=n_heads; self.d_k=d_k; self.d_v=d_k; self.device=device
        self.wq = nn.Linear(d_model, d_k*n_heads)
        self.wk = nn.Linear(d_model, d_k*n_heads)
        self.wv = nn.Linear(d_model, d_k*n_heads)
        self.fc = nn.Linear(n_heads*d_k, d_model)
        self.ln = nn.LayerNorm(d_model)
    def forward(self,Q,K,V,mask):
        bs = Q.size(0)
        q = self.wq(Q).view(bs,-1,self.n_heads,self.d_k).transpose(1,2)
        k = self.wk(K).view(bs,-1,self.n_heads,self.d_k).transpose(1,2)
        v = self.wv(V).view(bs,-1,self.n_heads,self.d_v).transpose(1,2)
        context,attn = ScaledDotProductAttention(self.d_k,self.device)(q,k,v,mask)
        context = context.transpose(1,2).contiguous().view(bs,-1,self.n_heads*self.d_v)
        out = self.fc(context)
        return self.ln(out + Q), attn

class PoswiseFFN(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__(); self.fc1=nn.Linear(d_model,d_ff); self.fc2=nn.Linear(d_ff,d_model)
    def forward(self,x): return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self,n_heads,d_model,d_ff,d_k,device):
        super().__init__(); self.attn=MultiHeadAttention(n_heads,d_model,d_k,device); self.ffn=PoswiseFFN(d_model,d_ff)
    def forward(self,x,mask): x2,_=self.attn(x,x,x,mask); return self.ffn(x2), _

class BERT(nn.Module):
    def __init__(self,n_layers,n_heads,d_model,d_ff,d_k,n_segments,vocab_size,max_len,device):
        super().__init__(); self.d_model=d_model; self.embedding=Embedding(vocab_size,max_len,n_segments,d_model,device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads,d_model,d_ff,d_k,device) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model,d_model); self.activ = nn.Tanh(); self.classifier = nn.Linear(d_model,2)
        self.linear = nn.Linear(d_model,d_model); self.norm = nn.LayerNorm(d_model)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device=device
    def forward(self,input_ids,segment_ids,masked_pos):
        out = self.embedding(input_ids, segment_ids)
        mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for l in self.layers: out,_ = l(out, mask)
        pooled = self.activ(self.fc(out[:,0])); logits_nsp = self.classifier(pooled)
        masked_pos = masked_pos[:,:,None].expand(-1,-1,out.size(-1))
        h_masked = torch.gather(out,1,masked_pos)
        h_masked = self.norm(torch.nn.functional.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        return logits_lm, logits_nsp
    def get_sentence_embedding(self,input_ids,segment_ids):
        out = self.embedding(input_ids, segment_ids)
        mask = get_attn_pad_mask(input_ids,input_ids,self.device)
        for l in self.layers: out,_ = l(out,mask)
        return out[:,0]

class SentenceBERT(nn.Module):
    def __init__(self, bert_model, d_model, num_classes=3, device='cpu'):
        super().__init__(); self.bert=bert_model; self.d_model=d_model; self.device=device
        self.classifier = nn.Sequential(nn.Linear(3*d_model, 2*d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2*d_model, num_classes))
    def forward(self, p_ids, p_seg, h_ids, h_seg):
        u = self.bert.get_sentence_embedding(p_ids, p_seg)
        v = self.bert.get_sentence_embedding(h_ids, h_seg)
        x = torch.cat([u,v,torch.abs(u-v)], dim=1)
        return self.classifier(x)
    def get_embeddings(self, ids, seg): return self.bert.get_sentence_embedding(ids, seg)

# Load models for all modes
modes = ['test', 'small', 'medium']
mode_configs = {
    'test': {'n_layers': 2, 'n_heads': 2, 'd_model': 128, 'd_ff': 512, 'd_k': 64},
    'small': {'n_layers': 4, 'n_heads': 4, 'd_model': 256, 'd_ff': 1024, 'd_k': 64},
    'medium': {'n_layers': 4, 'n_heads': 4, 'd_model': 256, 'd_ff': 1024, 'd_k': 64}
}
encoders = {}
sberts = {}

for mode in modes:
    config = mode_configs[mode]
    # Load vocab for each mode (assuming same vocab)
    vocab_size = vocab_config['vocab_size']
    max_len = vocab_config['max_len']
    
    # Instantiate models with correct parameters for each mode
    encoder = BERT(n_layers=config['n_layers'], n_heads=config['n_heads'], d_model=config['d_model'], 
                   d_ff=config['d_ff'], d_k=config['d_k'], n_segments=2, vocab_size=vocab_size, max_len=max_len, device=device)
    sbert = SentenceBERT(encoder, d_model=config['d_model'], device=device)
    
    # Load state dicts
    encoder.load_state_dict(torch.load(f'../models/bert_{mode}.pt', map_location=device))
    sbert.load_state_dict(torch.load(f'../models/sbert_{mode}.pt', map_location=device))
    
    encoder.eval()
    sbert.eval()
    
    encoders[mode] = encoder
    sberts[mode] = sbert

def tokenize_text(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

def predict_nli(premise, hypothesis):
    results = {}
    for mode in modes:
        sbert = sberts[mode]
        sbert.eval()
        def tok(t):
            words = tokenize_text(t)
            ids = [word2id.get(w, word2id['[UNK]']) for w in words]
            if len(ids) > max_len: ids = ids[:max_len]
            else: ids += [word2id['[PAD]']] * (max_len - len(ids))
            return torch.tensor([ids], dtype=torch.long).to(device)
        p_ids = tok(premise)
        h_ids = tok(hypothesis)
        with torch.no_grad():
            logits = sbert(p_ids, torch.zeros_like(p_ids), h_ids, torch.zeros_like(h_ids))
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred].item()
        results[mode] = {'prediction': ['entailment', 'neutral', 'contradiction'][pred], 'confidence': conf}
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    premise = data.get('premise', '')
    hypothesis = data.get('hypothesis', '')
    if not premise or not hypothesis:
        return jsonify({'error': 'Please provide both premise and hypothesis'}), 400
    result = predict_nli(premise, hypothesis)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)