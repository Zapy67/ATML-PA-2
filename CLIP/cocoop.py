# from torch.nn import Module
# import torch
# from torch import nn
# import clip

# class TextEncoder(Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype

#     def forward(self, prompts, tokenized_prompts):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # BND -> NBD
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # NBD -> BND
#         x = self.ln_final(x)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
#         return x

# class MetaNet(nn.Module):
#     def __init__(self, in_features, dtype):
#         super().__init__()
#         self.layer1 = nn.Linear(in_features, 32, dtype=dtype)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer2 = nn.Linear(32, in_features, dtype=dtype)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.relu(x)
#         x = self.layer2(x)
#         return x

# class PromptLearner(nn.Module):
#     def __init__(self, classes, clip_model, n_ctx):
#         super().__init__()
#         self.n_cls = len(classes)   
#         self.n_ctx = n_ctx  
#         self.ctx_dim = clip_model.ln_final.weight.shape[0]
#         dtype = clip_model.dtype

#         ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim).type(dtype)
#         nn.init.normal_(ctx_vectors, std=0.02)  # the coop implementation initializes vectors randomly
#         prompt_prefix = " ".join(["X"] * self.n_ctx)  # place holder for learnable tokens  
#         self.ctx = nn.Parameter(ctx_vectors)
#         self.metanet = MetaNet(clip_model.ln_final.weight.shape[0], dtype)

#         classnames = [name.replace("_", " ") for name in classes]
#         prompts = [prompt_prefix + " " + cls + "." for cls in classnames]
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
#         self.register_buffer("tokenized_prompts", tokenized_prompts)
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(self.tokenized_prompts).to(dtype)

#         self.register_buffer("prefix", embedding[:, :1, :]) # [SOS] token
#         self.register_buffer("suffix", embedding[:, 1 + self.n_ctx :, :]) # class token + [EOS] token
        

#     def forward(self, image_features):
#         ctx = self.ctx                      # (n_ctx, ctx_dim)
#         bias = self.metanet(image_features) # (batch, ctx_dim)
#         bias = bias.unsqueeze(1)            # (batch, 1, ctx_dim)
#         ctx = ctx.unsqueeze(0)              # (1, n_ctx, ctx_dim)
#         ctx_shifted = ctx + bias            # (batch, n_ctx, ctx_dim)
        
#         prompts = []
#         for ctx_shifted_i in ctx_shifted:
#             ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
#             pts_i = torch.cat([self.prefix, ctx_i, self.suffix], dim=1)
#             prompts.append(pts_i)
#         prompts = torch.stack(prompts)
        
#         return prompts
    
        
# class CustomCLIP(nn.Module):
#     def __init__(self, classnames, clip_model, n_ctx):
#         super().__init__()
#         self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts
#         self.image_encoder = clip_model.visual
#         self.text_encoder = TextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype

#     def forward(self, image_features):
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         prompts = self.prompt_learner(image_features)  # we normalize first before pushing true metanet for stability    
#         tokenized_prompts = self.tokenized_prompts
    
#         logits = []
#         for prompt, feats in zip(prompts, image_features):
#             text_features = self.text_encoder(prompt, tokenized_prompts)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#             logit = 100 * feats @ text_features.t()
#             logits.append(logit)
#         logits = torch.stack(logits)

#         return logits
    

# def train(cache, classnames, config, device):
#     lr = config.get('lr', 1e-5)
#     shot = config.get('shot', 1)
#     epochs = config.get('epochs', 5)
#     n_ctx = config.get('n_ctx', 16)
    
#     freq_count = {cls:0 for cls in classnames}
#     feats = []
#     labels = []
    
#     for feat, label in zip(*cache):
#         cls = classnames[label.item()]
#         if freq_count[cls]<shot:
#             feats.append(feat)
#             labels.append(label)
#             freq_count[cls] += 1
    
#     X = torch.stack(feats).float()
#     Y = torch.stack(labels)

#     clip_model = clip.load("ViT-B/32", device=device)
#     model = CustomCLIP(classnames, clip_model, n_ctx)
#     loss_fn = nn.CrossEntropyLoss()
#     dtype = model.dtype

#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#     model.to(device)
#     model.train()
#     X , Y = X.to(device).type(dtype), Y.to(device).long()
    
#     for _ in range(epochs):
#         optimizer.zero_grad()
#         logits = model(X)
#         loss = loss_fn(logits, Y)
#         loss.backward()
#         optimizer.step()
    
#     return model