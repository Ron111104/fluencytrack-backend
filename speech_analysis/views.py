import os
import numpy as np
import librosa
import tensorflow as tf
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import AudioFileSerializer
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = os.path.join(os.getcwd(), 'models', 'neurological_audio_classifier_final.keras')
model = tf.keras.models.load_model(model_path)
print(model_path)

# Initialize the label encoder
CLASSES = ['Alzheimer', 'Healthy', 'Parkinson']
label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)

n_mfcc = 13
max_pad_len = 200

def extract_mfcc(audio_file, n_mfcc=n_mfcc, max_pad_len=max_pad_len):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        return mfcc
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        return None

def predict_file(file_path):
    """Predicts the class of a single audio file using the trained model."""
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return "Could not process the file."
    
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    
    predictions = model.predict(mfcc)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]
    
    return predicted_class

@api_view(['POST'])
def analyze_audio(request):
    """Handles the uploaded audio file and returns the prediction."""
    print(request.data)
    serializer = AudioFileSerializer(data=request.data)
    if serializer.is_valid():
        audio_file = serializer.validated_data['file']
        
        # Save the file temporarily to process it
        file_path = os.path.join(os.getcwd(),'temp_audio', audio_file.name)
        print(file_path)
        with open(file_path, 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # Make prediction
        file_path2 = os.path.join(os.getcwd(),'temp_audio', 'adrso031.wav')
        predicted_class = predict_file(file_path)
        print(predicted_class)
        # Clean up by removing the temporary file
        # os.remove(file_path)
        
        return Response({'prediction': predicted_class})
    return Response(serializer.errors, status=400)
