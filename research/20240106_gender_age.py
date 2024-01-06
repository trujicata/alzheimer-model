# %%
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# %%
# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert (
            frames % frame_patch_size == 0
        ), "Frames must be divisible by frame patch size"

        num_patches = (
            (image_height // patch_height)
            * (image_width // patch_width)
            * (frames // frame_patch_size)
        )
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.age_embedding = nn.Embedding(51, dim)
        self.sex_embedding = nn.Embedding(2, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 3, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer3 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, video, age, sex):
        visual_output = self.to_patch_embedding(video)
        b, n, _ = visual_output.shape
        age_embedding = self.age_embedding(age)
        sex_embedding = self.sex_embedding(sex)

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, age_embedding, sex_embedding, visual_output), dim=1)
        x += self.pos_embedding[:, : (n + 3)]
        x = self.dropout(x)

        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# %%
model = ViT(
    image_size=100,
    image_patch_size=50,
    frames=120,
    frame_patch_size=20,
    num_classes=3,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
    pool="cls",
    channels=1,
    dim_head=64,
)

# %%
image = torch.randn(1, 1, 120, 100, 100)
age = torch.tensor([3]).unsqueeze(0)
sex = torch.tensor([1]).unsqueeze(0)

out = model(image, age, sex)
out.shape
# %%
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print("model size: {:.3f}MB".format(size_all_mb))

# %%
from models.classifier3D.model import Classifier3D

model = Classifier3D()
# %%
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print("model size: {:.3f}MB".format(size_all_mb))

# %%
import start
from models.vit.model import ViTClassifier3D
import yaml
import os

# %%
for config_file in os.listdir("models/vit/versions"):
    path = os.path.join("models/vit/versions", config_file)
    with open(path, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    model = ViTClassifier3D(
        name=args["model_name"],
        lr=args["lr"],
        image_patch_size=args["image_patch_size"],
        frame_patch_size=args["frame_patch_size"],
        dim=args["dim"],
        depth=args["depth"],
        heads=args["heads"],
        mlp_dim=args["mlp_dim"],
        pool=args["pool"],
        dropout=args["dropout"],
        emb_dropout=args["emb_dropout"],
        class_weights=args["class_weights"],
        weight_decay=args["weight_decay"],
        scheduler_step_size=args["scheduler_step_size"],
        scheduler_gamma=args["scheduler_gamma"],
    )
    with torch.no_grad():
        out = model(
            torch.randn(1, 1, 120, 100, 100),
            torch.tensor([3]).unsqueeze(0),
            torch.tensor([1]).unsqueeze(0),
        )
    print(out.shape)
    del model

# %%
from models.vit.dataset import ADNIDataModule

datamodule = ADNIDataModule("data/original", batch_size=8)
datamodule.setup("fit")
# %%
model = ViTClassifier3D("hola")

# %%
batch = next(iter(datamodule.train_dataloader()))
# %%
model.training_step(batch, 0)

# %%
