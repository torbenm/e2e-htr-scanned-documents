import tensorflow as tf
import os
from nn.layer.tf_unet import unet
from data.PaperNoteSlices import PaperNoteSlices


class DataProvider(object):

    def __init__(self):
        self.data = PaperNoteSlices(slice_width=512, slice_height=512)
        self.val_call = True
        self.generator = None

    def __call__(self, batch_size):
        if self.val_call:
            vals = self.data.generateBatch(
                batch_size=batch_size, dataset="dev")
            X, Y, _ = next(vals)
            self.val_call = False
        else:
            if self.generator is None:
                self.generator = self.data.generateBatch(
                    batch_size=batch_size, dataset="train")
            try:
                X, Y, _ = next(vals)
            except StopIteration:
                self.generator = None
                return self(batch_size)
        return X/255.0, Y/255.0


os.environ["CUDA_VISIBLE_DEVICES"] = str(3)

with tf.device("/device:GPU:3"):

    data_provider = DataProvider()

    net = unet.Unet(channels=1,
                    n_class=2,
                    layers=3,
                    features_root=64,
                    cost_kwargs=dict(regularizer=0.001),
                    )

    trainer = unet.Trainer(net, optimizer="momentum",
                           opt_kwargs=dict(momentum=0.2))
    path = trainer.train(data_provider, "./tfunet_ex",
                         training_iters=32,
                         epochs=1,
                         dropout=0.5,
                         display_step=2)
