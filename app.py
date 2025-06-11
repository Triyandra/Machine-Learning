import discord
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Inisialisasi client Discord
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Load dataset
data = pd.read_csv('heart.csv')

# Preprocessing (ambil fitur dan target)
X = data.drop('target', axis=1)
y = data['target']

# Train model (untuk contoh, training di sini)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# (Opsional) Simpan model
joblib.dump(model, 'model_heart.pkl')

# Load model
model = joblib.load('model_heart.pkl')

@client.event
async def on_ready():
    print(f'Bot sudah online! Login sebagai: {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!prediksi'):
        try:
            # Format: !prediksi umur cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal
            # Contoh: !prediksi 63 3 145 233 1 0 150 0 2.3 0 0 1
            data_input = message.content.split()[1:]
            data_input = [float(i) for i in data_input]
            features = np.array(data_input).reshape(1, -1)

            # Prediksi
            prediction = model.predict(features)
            result = "POTENSI PENYAKIT JANTUNG" if prediction[0] == 1 else "TIDAK TERDETEKSI"

            await message.channel.send(f'Hasil prediksi: {result}')

        except Exception as e:
            await message.channel.send(f'Format salah atau error: {e}')

# Token bot (ganti dengan token bot Discord Anda)
TOKEN = 'YOUR_DISCORD_BOT_TOKEN'
client.run(TOKEN)
