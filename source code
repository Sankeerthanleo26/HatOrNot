import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

# Load InceptionV3 model without top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add new layers for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  
predictions = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Path to the test set folder containing images
test_set_folder = '/Users/sankeerthanleo/Desktop/test_set/test_set'

# Make predictions on test images
predictions = []

for filename in os.listdir(test_set_folder):
    if filename.endswith('.jpg'):
        img_path = os.path.join(test_set_folder, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0) 

        prediction = model.predict(img_array)
        predicted_class = 'Hat' if prediction >= 0.5 else 'No Hat'

        photo_id = os.path.splitext(filename)[0]
        predictions.append({'id': photo_id, 'class': predicted_class})

# Create submission file
submission_df = pd.DataFrame(predictions, columns=['id', 'class'])
submission_df.to_csv('/Users/sankeerthanleo/Desktop/sb.csv', index=False)
