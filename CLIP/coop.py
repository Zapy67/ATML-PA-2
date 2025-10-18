from torch.nn import Module
import torch
from torch import nn
import clip

class TextEncoder(Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  #BNE -> NBE
        x = self.transformer(x)
        x = x.permute(1, 0, 2) #BNE -> NBE 
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(Module):
    def __init__(self, classes, clip_model, csc, n_ctx):
        super().__init__()
        self.n_cls = len(classes)   
        self.n_ctx =  n_ctx  
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype

        if csc:
            ctx_vectors = torch.empty(self.n_cls, self.n_ctx, self.ctx_dim).type(dtype)
        else:
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim).type(dtype)
        
        nn.init.normal_(ctx_vectors, std=0.02) #the coop implementation intializes vectors randomly
        prompt_prefix = " ".join(["X"] * self.n_ctx)  #place holder for learnable tokens  
        self.ctx = nn.Parameter(ctx_vectors) 

        classnames = [name.replace("_", " ") for name in classes]
        prompts = [prompt_prefix + " "+cls + "." for cls in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).type(dtype)

        self.register_buffer("prefix", embedding[:, :1, :])  #[SOS] token
        self.register_buffer("suffix", embedding[:, 1 + self.n_ctx :, :])  #class token + [EOS] token

        
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) #expands the ctx to the appropriate view, when dealing with unified context.
            #(self.n_cls, n_ctx, ctx_dim)
     
        prompts = torch.cat(
            [self.prefix,ctx,self.suffix,],
            dim=1,
        )
        return prompts
        
class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model, csc, n_ctx):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, csc, n_ctx)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_features):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100 * image_features @ text_features.t()

        return logits
    

def train(cache, classnames, config, device):
    lr = config.get('lr', 1e-5)
    shot = config.get('shot', 1)
    epochs = config.get('epochs', 5)
    csc = config.get('csc', False)
    n_ctx = config.get('n_ctx', 16)
    
    freq_count = {cls:0 for cls in classnames}
    feats = []
    labels = []
    
    for feat, label in zip(*cache):
        cls = classnames[label.item()]
        if freq_count[cls]<shot:
            feats.append(feat)
            labels.append(label)
            freq_count[cls] += 1
    
    X = torch.stack(feats).float()
    Y = torch.stack(labels)
    
    clip_model = clip.load('"ViT-B/32"', device=device)
    model = CustomCLIP(classnames, clip_model, csc, n_ctx)
    loss_fn = nn.CrossEntropyLoss()
    dtype = model.dtype

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    X , Y = X.to(device).type(dtype), Y.to(device).long()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y)
        loss.backward()
        optimizer.step()
    
    return model

    