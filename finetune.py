import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go
from datasets import load_dataset
from gensim import downloader
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.utils import tokenize as gensim_tokenize
from scipy.stats import spearmanr
from tqdm import trange

test_dataset = load_dataset("stsb_multi_mt", "en", split="test")
train_dataset = load_dataset("stsb_multi_mt", "en", split="train")
dev_dataset = load_dataset("stsb_multi_mt", "en", split="dev")

glove_model = downloader.load("glove-twitter-25")
pad_index = glove_model.add_vector(
    "[PAD]", np.full(glove_model.vector_size, np.nan)
)

embeddings = glove_model.vectors
vocab = glove_model.index_to_key


def tokenize_text(text: str) -> list[int]:
    tokens = gensim_tokenize(text, lowercase=True, deacc=True)
    token_ids = [glove_model.get_index(token, pad_index) for token in tokens]
    return token_ids


def tokenize_example(example):
    example["sentence1"] = tokenize_text(example["sentence1"])
    example["sentence2"] = tokenize_text(example["sentence2"])
    return example


def pad(ids: list[int], desired_length: int):
    ids = ids[:desired_length]
    pad_len = desired_length - len(ids)
    if pad_len > 0:
        ids = ids + [pad_index] * pad_len
    return ids


def pad_example(example, desired_length: int):
    example["sentence1"] = pad(example["sentence1"], desired_length)
    example["sentence2"] = pad(example["sentence2"], desired_length)
    return example


def forward(x1, x2, sim, embeddings, bias, slope):
    pooled1 = jnp.nanmean(jnp.take(embeddings, x1, axis=0), axis=1)
    pooled2 = jnp.nanmean(jnp.take(embeddings, x2, axis=0), axis=1)
    cosine_sim = optax.cosine_similarity(pooled1, pooled2)
    scaled_sim = slope * cosine_sim + bias
    return optax.squared_error(scaled_sim, sim).mean()


def loss_func(params, x1, x2, sim):
    return forward(x1, x2, sim, **params)


def split_by_batch_size(arr, batch_size):
    nbatches = arr.shape[0] // batch_size
    if nbatches != arr.shape[0] / batch_size:
        nbatches += 1
    return jnp.array_split(arr, nbatches)


def prepare_data(ds, max_length=128) -> dict:
    ds = ds.map(tokenize_example)
    ds = ds.map(lambda ex: pad_example(ex, desired_length=max_length))
    return dict(
        x1=jnp.array(np.stack(ds["sentence1"])),
        x2=jnp.array(np.stack(ds["sentence2"])),
        sim=jnp.array(ds["similarity_score"]),
    )


def take_batch(data: dict, idx: np.ndarray) -> dict:
    return dict(x1=data["x1"][idx], x2=data["x2"][idx], sim=data["sim"][idx])


def evaluate_spearmanr(embeddings, x1, x2, sim):
    pooled1 = jnp.nanmean(jnp.take(embeddings, x1, axis=0), axis=1)
    pooled2 = jnp.nanmean(jnp.take(embeddings, x2, axis=0), axis=1)
    pred_sim = optax.cosine_similarity(pooled1, pooled2)
    res = spearmanr(pred_sim, sim)
    return res.statistic


@jax.jit
def step(params, opt_state, batch):
    loss_value, grads = jax.value_and_grad(loss_func)(params, **batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


train_data = prepare_data(train_dataset)
test_data = prepare_data(test_dataset)
dev_data = prepare_data(dev_dataset)

batch_size = 512
learning_rate = 0.005
n_epochs = 50
train_loss = []
val_loss = []

key = jax.random.PRNGKey(0)
params = dict(embeddings=jnp.array(embeddings), bias=1.0, slope=2.5)
optimizer = optax.adam(learning_rate=learning_rate)
opt_state = optimizer.init(params)
batch_indices = jnp.arange(len(train_data["x1"]))
key, sub = jax.random.split(key)
batch_indices = jax.random.shuffle(sub, batch_indices)
batch_indices = split_by_batch_size(batch_indices, batch_size)
# Initial loss
train_loss.append(loss_func(params, **train_data))
val_loss.append(loss_func(params, **train_data))
for epoch in trange(n_epochs):
    for batch_idx in batch_indices:
        batch = take_batch(train_data, batch_idx)
        params, opt_state, loss_value = step(params, opt_state, batch)
    train_loss.append(loss_func(params, **train_data))
    val_loss.append(loss_func(params, **dev_data))


epochs = np.arange(n_epochs)
fig = go.Figure()
fig = fig.add_scatter(x=epochs, y=train_loss, name="train_loss")
fig = fig.add_scatter(x=epochs, y=val_loss, name="val_loss")
fig.show()

print("Spearman R2 scores for sentence similarity task:")
print(
    " - Original model: ", evaluate_spearmanr(glove_model.vectors, **test_data)
)
print(
    " - Finetuned model:",
    evaluate_spearmanr(params["embeddings"], **test_data),
)

finetuned_vectors = KeyedVectors(glove_model.vector_size)
finetuned_vectors.add_vectors(vocab, params["embeddings"])
finetuned_vectors.save("finetuned_glove.model")
