Fun visualization of the 2D latent space of an autoencoder trained on Fashion-MNIST. Inspired by and credits to n8python for the js code

Interactive JS implementation available on my github page

---

Local replication instructions :

- All the training code is available in `training` dir

- Once your model are trained, convert them to TensorflowJS using `tensorflowjs_converter --input_format=keras /saved_models/best_model.h5 /tf_js/...` (if you trained it using Keras Functional API, you may need to change the `modelTopology.model_config.class_name` attribute of `model.json` from `Functional` to `Model`)

- You also need to generate the map using the encoder by executing the `GenerateMap` notebook

- Then you can visualize the latent space of the AE by executing the `mapSpace.py` script (Tkinter GUI) 

---

Discussion : below are two screenshots of the train / test distribution of the current online implementation. Despite having a custom loss acting as a penalty to the density, there are still some "blind spots" in the latent space. I think it's possible to do better but requires some fine-tuning

![Train latent space density](/img/train_latent.png)
![Test latent space density](/img/test_latent.png)

Developing the same approach on more complex datasets such as CIFAR-10 could also be fun, but compressing it to 2D may be a bit too "extreme", thus the results may not be as visually interesting