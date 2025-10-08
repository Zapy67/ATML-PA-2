import deeplake

sources = ["Art", "Realworld", "Product"]
target = "Clipart"

ds = deeplake.open('hub://activeloop/office-home-domain-adaptation')

# source_mask = [d in sources for d in ds["domain"].numpy()]
# target_mask = [d == target for d in ds["domain"].numpy()]

# source_ds = ds[source_mask]
# target_ds = ds[target_mask]

# source_loader = source_ds.pytorch(num_workers = 0, batch_size= 4, shuffle = False, drop_last=True)
# target_loader = target_ds.pytorch(num_workers = 0, batch_size= 4, shuffle = False, drop_last=True)




