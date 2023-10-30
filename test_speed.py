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

print("jax local devices:",jax.local_devices())
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.ERROR)

# Define EncoderDecoderModel constructor args (except the module).
# input_vocabulary=t5.data.get_default_vocabulary()
# output_vocabulary=t5.data.get_default_vocabulary()
VOCABULARY = seqio.SentencePieceVocabulary("gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model")
optimizer=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0, logical_factor_rules=t5x.adafactor.standard_logical_factor_rules())


# The checkpoint below is a T5-1.1-Small checkpoint (https://github.com/google-research/t5x/blob/main/docs/models.md)
# that has additionally been finetuned on the (Open Domain) Natural Questions
# benchmark (https://ai.google.com/research/NaturalQuestions).
checkpoint_path='gs://t5-data/pretrained_models/t5x/t5_base/checkpoint_999900'
dtype='bfloat16'
restore_mode='specific'


partitioner=partitioning.PjitPartitioner(
        num_partitions=1,
        model_parallel_submesh=None)

with open("data_for_speedtest.json") as f:
    examples_of_lengths = json.load(f)
start_t = time.time()
print("mark:start!")
last_t = start_t
    
batch_size=1
# res = []
for modelname in ["t5-linformer","t5-vanilla"]:
    for length in examples_of_lengths:
        if int(length) < 40000 : continue
        examples = [examples_of_lengths[length]]*batch_size
        print("length sum:",length)
        input_length = len(examples[0]["input"].split(" "))
        target_length=len(examples[0]["target"].split(" "))
        decode_fn=functools.partial(t5x.decoding.temperature_sample, temperature=1.0, topk=40,max_decode_steps = 1)

        # Define a model using the minimal T5 module.
        t5_module = network.Transformer(config=network.T5Config(
            vocab_size=32128,
            dtype='bfloat16',
            emb_dim=768,
            num_heads=12,
            num_encoder_layers=12,
            num_decoder_layers=12,
            head_dim=64,
            mlp_dim=3072,
            mlp_activations=('relu',),
            dropout_rate=0.0,
            logits_via_embedding=True,
            linformer= (modelname=="t5-linformer")
            ))
        model = t5x.models.EncoderDecoderModel(
            module=t5_module,
            input_vocabulary=VOCABULARY,
            output_vocabulary=VOCABULARY,
            optimizer_def=optimizer,
            decode_fn=decode_fn)
        
        task_feature_lengths = {'inputs': input_length, 'targets': target_length}
        output_dir='/tmp/output_dir'
        input_shapes = {
            'encoder_input_tokens': np.array([batch_size, input_length]),
            'decoder_target_tokens': np.array([batch_size, target_length]),
            'decoder_input_tokens': np.array([batch_size, target_length]),
            'decoder_loss_weights': np.array([batch_size, target_length])
        }

        interactive_model = InteractiveModel(
        batch_size=batch_size,
        task_feature_lengths=task_feature_lengths,
        output_dir=output_dir,
        partitioner=partitioner,
        model=model,
        dtype=dtype,
        restore_mode=restore_mode,
        checkpoint_path=checkpoint_path,
        input_shapes=input_shapes,
        from_scratch = True
        )
        
        for i in range(2): 
            # print("input:",examples[0]['input'])
            # print("target_length:",target_length)
            examples_and_predictions, _, infer_time = block_until_ready(interactive_model.predict_with_aux(examples=examples))
            predictions = [prediction for example, prediction in examples_and_predictions]
            print(f"Inference batch {i} complete, use time: {infer_time}")
            output_len = len(predictions[0].decode().split(" "))
            print ("output len: ",output_len)
            # res.append({"inlen":input_length,"outlen":output_len,"time":time.time()-last_t})
            with open("res.json","a")as f:
                f.write("\n")
                f.write(json.dumps({"model":modelname,"inlen":input_length,"outlen":output_len,"time":infer_time},ensure_ascii =False))
            last_t = time.time()
end_t = time.time()
print("mark:end!")
# predictions = [prediction for example, prediction in examples_and_predictions]
# print(f"Predictions: {predictions}\n")
print("Inference time:",end_t-start_t)
# with open("res.json","a")as f:
#     f.write(json.dumps(res,ensure_ascii =False))
