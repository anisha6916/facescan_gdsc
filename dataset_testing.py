# import pandas as pd
# import numpy as np
# import seaborn as sns
# import os
# from PIL import Image, ImageOps
# from sklearn.model_selection import train_test_split

# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
# from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf


# images = []
# ages = []
# genders = []

# for i in os.listdir('../input/utkface-new/crop_part1/')[0:8000]:
#     split = i.split('_')
#     ages.append(int(split[0]))
#     genders.append(int(split[1]))
#     images.append(Image.open('../input/utkface-new/crop_part1/' + i))

# images = pd.Series(list(images), name = 'Images')
# ages = pd.Series(list(ages), name = 'Ages')
# genders = pd.Series(list(genders), name = 'Genders')

# df = pd.concat([images, ages, genders], axis=1)
# df

# display(df['Images'][0])
# print(df['Ages'][0], df['Genders'][0])


import pandas as pd
import numpy as np
import seaborn as sns
import os
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# data path
image_directory = #'/Users/leebyung2/Desktop/facescan_gdsc-main/crop_part1/'

# ex) 28_1_2_20170116164204569.jpg.chip
#####################################
# 28: age
# 0: Male, 1: Female
# 0: White, 1: Black, 2: Asian, 3: Indian, 4: Other races 
#####################################


images = []
ages = []
#genders = []
#races = []


for file_name in os.listdir(image_directory)[:8000]:
    try:
        split = file_name.split('_')
        age = int(split[0])
        #gender = int(split[1]) 
        #race = int(split[2]) 
        img = Image.open(os.path.join(image_directory, file_name)).resize((128, 128)).convert('RGB')  # resize for faster training of models
        images.append(np.array(img))
        ages.append(age)
        #genders.append(gender)
        #races.append(race)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        continue

# data to NumPy array
images = np.array(images)
ages = np.array(ages)
# genders = np.array(genders)
# races = np.array(races)

# data split
X_train, X_test, y_train_age, y_test_age = train_test_split(images, ages, test_size=0.2, random_state=42)

# nomalizing
X_train = X_train / 255.0
X_test = X_test / 255.0

# age model
age_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

# model compile
age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# training the model
age_model.fit(X_train, y_train_age, epochs=10211)



# def predict

def predict_age(img_path):
    img = Image.open(img_path).resize((128, 128)).convert('RGB')  # include color channel?
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predicted_age = age_model.predict(img_array)[0][0]
    
    return predicted_age

# image evaluate
test_img_path = # '/Users/leebyung2/Desktop/facescan_gdsc-main/test_image1.png'
predicted_age = predict_age(test_img_path)
print(f"Predicted Age : {predicted_age}")