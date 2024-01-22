import functools
import os
import time
import json
import clu.data.dataset_iterator
import tensorflow as tf
import jax
from jax import random
from jax import block_until_ready
from jax.experimental import multihost_utils
import jax.numpy as jnp
from flax import linen
import numpy as np
import seqio
import t5.data
from t5.evaluation import metrics as t5_metrics
import t5x
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import utils
from t5x.examples.t5 import network
from t5x.examples.scalable_t5 import network as scalable_network
from t5x.interactive_model import InteractiveModel
from t5x.interactive_model import get_batches_from_seqio
from t5x.interactive_model import InferenceType
import logging
from jax.config import config
from tqdm import tqdm
# config.update('jax_disable_jit', True)

print("jax local devices:", jax.local_devices())
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.ERROR)

# Define EncoderDecoderModel constructor args (except the module).
# input_vocabulary=t5.data.get_default_vocabulary()
# output_vocabulary=t5.data.get_default_vocabulary()
VOCABULARY = seqio.SentencePieceVocabulary(
    "gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model")
optimizer = t5x.adafactor.Adafactor(
    decay_rate=0.8, step_offset=0, logical_factor_rules=t5x.adafactor.standard_logical_factor_rules())


# The checkpoint below is a T5-1.1-Small checkpoint (https://github.com/google-research/t5x/blob/main/docs/models.md)
# that has additionally been finetuned on the (Open Domain) Natural Questions
# benchmark (https://ai.google.com/research/NaturalQuestions).
checkpoint_path = 'gs://t5-data/pretrained_models/t5x/t5_base/checkpoint_999900'
dtype = 'bfloat16'
restore_mode = 'specific'


partitioner = partitioning.PjitPartitioner(
    num_partitions=1,
    model_parallel_submesh=None)

# with open("data_for_speedtest.json") as f:
#     examples_of_lengths = json.load(f)
start_t = time.time()
print("mark:start!")
last_t = start_t

batch_size = 1
# res = []
for modelname in ["lin+ker", "kernel", "linformer", "vanilla"]:
# for modelname in ["t5-vanilla"]:
    for l in tqdm(range(12000,20000,2000)):
        # for length in ['1072']:
        # examples = [examples_of_lengths[length]]*batch_size
        print("input and output length:", l)
        # input_length = len(examples[0]["input"].split(" "))
        # target_length=len(examples[0]["target"].split(" "))
        #greedy 
        decode_fn = functools.partial(t5x.decoding.temperature_sample, temperature=0, topk=40, max_decode_steps=1
                                      )

        # Define a model using the minimal T5 module.
        t5_module = network.Transformer(config=network.T5Config(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=2600,
            num_heads=8,
            num_encoder_layers=12,
            num_decoder_layers=12,
            head_dim=64,
            mlp_dim=3072,
            mlp_activations=('relu',),
            dropout_rate=0.0,
            logits_via_embedding=True,
            linformer=("lin" in modelname),
            linformer_dim = 1500,
            kernel_method=('performer' if "ker" in modelname else None)
        ))
        model = t5x.models.EncoderDecoderModel(
            module=t5_module,
            input_vocabulary=VOCABULARY,
            output_vocabulary=VOCABULARY,
            optimizer_def=optimizer,
            decode_fn=decode_fn)
        decoder_len = l
        fake_encoder_inputs = jnp.ones((1, l)).astype(jnp.int32)
        fake_decoder_inputs = jnp.ones((1, decoder_len)).astype(jnp.int32)
        batch_shape ={'encoder_input_tokens': fake_encoder_inputs.shape,
                    'decoder_input_tokens': fake_decoder_inputs.shape}
        fake_batch = {'encoder_input_tokens': fake_encoder_inputs,
                    'decoder_input_tokens': fake_decoder_inputs}
        
        print("initializing model")
        # variables = t5_module.init(rngs= jax.random.PRNGKey(42),encoder_input_tokens=fake_encoder_inputs,decoder_input_tokens=fake_decoder_inputs,decoder_target_tokens=fake_decoder_inputs)
        variables = model.get_initial_variables(
            jax.random.PRNGKey(42),
            batch_shape
        )
        # _, initial_variables = t5_module.apply(
        #         {'params': variables['params']},
        #         encoder_input_tokens=fake_encoder_inputs,
        #         decoder_input_tokens=fake_decoder_inputs,
        #         decoder_target_tokens=fake_decoder_inputs,
        #         mutable=['cache'],
        #         decode=True,
        #         enable_dropout=False,
        #         decode_positional_encoding_index=0
        #     )
        # cache = initial_variables['cache']
        print("initialized.")
        # variables['cache']=cache
        # flat_ids = jnp.ones((1, 1))
        # encode_jit = jax.jit(functools.partial(
        #     t5_module.apply,
        #     method=network.Transformer.encode
        # ))
        # decode_jit = jax.jit(functools.partial(
        #     t5_module.apply,
        #     method=network.Transformer.decode,
        #     decode=True,
        #     mutable=['cache']
        # ))
        infer_jit = jax.jit(
            functools.partial(
                model.predict_batch_with_aux,
                params=variables['params'],
                prompt_with_targets=False
            )
        )
        for repeat in range(30):
            stime = time.time()
            # encoded = block_until_ready(
            #     encode_jit(variables, fake_encoder_inputs))
            # res = block_until_ready(decode_jit(
            #     variables, encoded, fake_encoder_inputs, flat_ids, flat_ids))
            res = block_until_ready(infer_jit(batch=fake_batch))
            # print(res)
            infer_time = time.time()-stime
            with open("res.json", "a")as f:
                f.write("\n")
                f.write(json.dumps({"model": modelname, "inlen": l,
                        "outlen": decoder_len, "time": infer_time}, ensure_ascii=False))
