#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
import numpy as np


# In[10]:


celsius_q    = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_q):
  print("{} градусів Цельсія = {} градусів Фаренгейта".format(c, fahrenheit_a[i]))


# In[13]:


model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Завершили тренировку моделі")


# In[14]:


print(model.predict([100.0]))


# In[ ]:




