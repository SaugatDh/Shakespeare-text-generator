# Shakespeare Text Generator

This document explains the code from the `Shakespeare_Text_generator.ipynb` notebook.

## 1. Download the dataset

Downloads the Tiny Shakespeare dataset from a URL using `tf.keras.utils.get_file` and reads it into a string variable `shakespeare_text`.

```python
import tensorflow as tf

shakespear_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filepath = tf.keras.utils.get_file("shakespeare.txt",shakespear_url)
with open(filepath) as f:
  shakespeare_text = f.read()
```

## 2. Preview the data

Prints the first 80 characters of the downloaded Shakespeare text to get a preview of the data.

```python
print(shakespeare_text[:80])
```

## 3. Tokenize the text

Initializes a `TextVectorization` layer for character-level tokenization, adapts it to the Shakespeare text to build the vocabulary, and then encodes the text into a sequence of character IDs.

```python
text_vec_layer = tf.keras.layers.TextVectorization(split="character",standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]
```

## 4. Adjust the encoded text

Adjusts the encoded character IDs by subtracting 2 and calculates the number of unique tokens in the vocabulary after the adjustment.

```python
encoded -= 2
n_tokens = text_vec_layer.vocabulary_size() -2
```

## 5. Create a dataset of windows

Defines a function `to_dataset` that takes a sequence of character IDs and converts it into a TensorFlow Dataset of input/target window pairs for training a sequence model.

```python
def to_dataset(sequence,length,shuffle=False,seed=None,batch_size=32):
  ds = tf.data.Dataset.from_tensor_slices(sequence)
  ds = ds.window(length + 1,shift=1,drop_remainder=True)
  ds = ds.flat_map(lambda window_ds:window_ds.batch(length+1))
  if shuffle:
    ds = ds.shuffle(buffer_size=100_00,seed=seed)
  ds = ds.batch(batch_size)
  return ds.map(lambda window: (window[:,:-1],window[:,1:])).prefetch(1)
```

## 6. Create training, validation, and test sets

Sets the window length for the dataset and splits the encoded text into training, validation, and test sets using the `to_dataset` function with specified sizes and shuffling for the training set.

```python
length =100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1_000_000],length=length,shuffle=True,seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000],length=length)
test_set = to_dataset(encoded[1_060_000:],length=length)
```

## 7. Build and compile the model

Builds a character-level RNN model using a GRU layer for sequence processing, followed by a Dense layer with softmax activation for character prediction. Compiles the model with sparse categorical crossentropy loss and Nadam optimizer, and sets up a ModelCheckpoint callback to save the best model based on validation accuracy during training.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens,output_dim=16),
    tf.keras.layers.GRU(128,return_sequences=True),
    tf.keras.layers.Dense(n_tokens,activation="softmax")
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])

# model_ckpt = tf.keras.callbacks.ModelCheckpoint(
#     "shakespeare_model.keras",monitor ="val_accuracy",save_best_only=True
# )

# history = model.fit(train_set,validation_data=valid_set,epochs=10,callbacks=[model_ckpt])
```

## 8. Use a pretrained model

As training takes a lot of time, we are using the pretrained model of shakespeare RNN model.

```python
import pathlib
import tensorflow as tf
```

```python
shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])
```

```python
url = "https://github.com/ageron/data/raw/main/shakespeare_model.tgz"
path = tf.keras.utils.get_file("shakespeare_model.tgz",url,extract=True)
model_path = pathlib.Path(path).with_suffix('').joinpath('shakespeare_model')
```

```python
model_path
```

```python
loaded_model = tf.keras.Sequential([
    tf.keras.layers.TFSMLayer(
        model_path,
        call_endpoint='serving_default'
    )
])
```

```python
loaded = tf.saved_model.load(model_path)
```

```python
print("Available signatures:", list(loaded.signatures.keys()))
```

```python
infer = None
if "serving_default" in loaded.signatures:
    infer = loaded.signatures["serving_default"]
else:
    # fallback to direct call
    infer = loaded.__call__
```

## 9. Generate new text

Now that we have the loaded model and can predict the next character, we can write a function to generate a sequence of text.

```python
def next_char(text,temperature=1):
  outputs = infer(text_vectorization_input=tf.constant([text]))
  logits = outputs['sequential'][:, -1, :] # Select logits for the last character
  rescaled_logits = logits / temperature # Use logits directly for temperature scaling
  char_id = tf.random.categorical(rescaled_logits,num_samples=1)[0,0]
  return text_vec_layer.get_vocabulary()[char_id.numpy()+2] # Convert char_id to numpy scalar
```

```python
def generate_text(text, n_chars=200, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text
```

```python
tf.random.set_seed(42)
print(generate_text("To be or not to be",temperature=0.01))
```

## 10. Create a Gradio Interface

This part of the code creates a simple web interface using the Gradio library to interact with the text generation model.

```python
!pip install gradio -q
import gradio as gr

def gradio_generate_text(prompt, n_chars=200, temperature=1):
    # Reuse the existing generate_text function
    return generate_text(prompt, n_chars, temperature)

# Define the Gradio interface
interface = gr.Interface(
    fn=gradio_generate_text,
    inputs=[
        gr.Textbox(label="Prompt", value="To be or not to be"),
        gr.Slider(minimum=50, maximum=500, value=200, label="Number of characters to generate"),
        gr.Slider(minimum=0.01, maximum=2.0, value=1.0, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10), # Added lines parameter to increase height
    title="Shakespearean Text Generator",
    description="Generate text in the style of Shakespeare using a trained RNN model."
)

# Launch the interface
interface.launch()
```
