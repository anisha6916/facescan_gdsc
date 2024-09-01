import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2 



# ex) 28_1_2_20170116164204569.jpg.chip
#####################################
# 28: age
# 0: Male, 1: Female
# 0: White, 1: Black, 2: Asian, 3: Indian, 4: Other races 
#####################################


image_directory =  r'gdsc/archive-3 copy/crop_part1'

valid_image = ['.jpg']

race_counts = {
    'White': 0,
    'Black': 0,
    'Asian': 0,
    'Indian': 0,
    'Other': 0
}


# # check the files in the folder to count data for each race 
# for file_name in os.listdir(image_directory):
#     if any(file_name.lower().endswith(ext) for ext in valid_image):
#         try:
#             split = file_name.split('_')
#             race = int(split[2])
#             if race == 0:
#                 race_counts['White'] += 1
#             elif race == 1:
#                 race_counts['Black'] += 1
#             elif race == 2:
#                 race_counts['Asian'] += 1
#             elif race == 3:
#                 race_counts['Indian'] += 1
#             elif race == 4:
#                 race_counts['Other'] += 1
#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")
#             continue
# for race, count in race_counts.items():
#     print(f"{race}: {count} images")

## Result
# White: 5265 images
# Black: 405 images
# Asian: 1553 images
# Indian: 1452 images
# Other: 1103 images


images = []
ages = []
genders = []
races = []

for file_name in os.listdir(image_directory)[:8000]:
    if any(file_name.lower().endswith(ext) for ext in valid_image):
        try:
            split = file_name.split('_')
            age = int(split[0])
            gender = int(split[1])
            race = int(split[2])
            img = Image.open(os.path.join(image_directory, file_name)).resize((128, 128)).convert('RGB')
            images.append(np.array(img))
            ages.append(age)
            genders.append(gender)
            races.append(race)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

# data to NumPy array
images = np.array(images)
ages = np.array(ages)
genders = np.array(genders)
races = np.array(races)

# data split
X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender, y_train_race, y_test_race = train_test_split(
    images, ages, genders, races, test_size=0.2, random_state=42)

# nomalizing
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the input layer
input_img = Input(shape=(128, 128, 3))

# Shared layers
x = Conv2D(32, (3, 3), activation='relu')(input_img)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Age prediction output
age_output = Dense(1, name='age')(x)

# Gender prediction output
gender_output = Dense(1, activation='sigmoid', name='gender')(x)

# Race prediction output
race_output = Dense(5, activation='softmax', name='race')(x)

# Define the model
model = Model(inputs=input_img, outputs=[age_output, gender_output, race_output])

# Compile the model
model.compile(optimizer='adam', 
              loss={'age': 'mse', 'gender': 'binary_crossentropy', 'race': 'categorical_crossentropy'},
              metrics={'age': 'mae', 'gender': 'accuracy', 'race': 'accuracy'})

# Convert race to categorical
y_train_race = tf.keras.utils.to_categorical(y_train_race, num_classes=5)
y_test_race = tf.keras.utils.to_categorical(y_test_race, num_classes=5)

# Training the model
model.fit(X_train, {'age': y_train_age, 'gender': y_train_gender, 'race': y_train_race}, epochs=10, 
          validation_data=(X_test, {'age': y_test_age, 'gender': y_test_gender, 'race': y_test_race}))

# Prediction function
def predict_attributes(image_array):
    #img = Image.open(img_path).resize((128, 128)).convert('RGB')
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)).resize((128, 128)).convert('RGB')

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predicted_age, predicted_gender, predicted_race = model.predict(img_array)
    
    predicted_gender = 'Male' if predicted_gender[0][0] < 0.5 else 'Female'
    predicted_race = np.argmax(predicted_race, axis=1)[0]
    
    races = ['White', 'Black', 'Asian', 'Indian', 'Other']
    predicted_race = races[predicted_race]
    
    return predicted_age[0][0], predicted_gender, predicted_race

#############################################################################################################
########################________________open cv______________________#########################################



# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def open_camera():
    capture = cv2.VideoCapture(0)
    
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rgb_face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            age, gender, race = predict_attributes(rgb_face)
            
            cv2.putText(frame, f'Age: {age:.2f}', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Gender: {gender}', (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Race: {race}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()

# Run the camera function
open_camera()