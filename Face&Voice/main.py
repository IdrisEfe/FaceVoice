import cv2
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from pvrecorder import PvRecorder
import wave # WAV (Sıkıştırılmamış)
import struct # decoding ve encoding (ses verilerini işlemek için (sesler binary şeklinde tutulur))
import datetime # Süre olarak ayarlayacağız

# cortana

face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
processor = AutoProcessor.from_pretrained('m3hrdadfi/wav2vec2-large-xlsr-turkish')

model = AutoModelForCTC.from_pretrained('m3hrdadfi/wav2vec2-large-xlsr-turkish')
recorder = PvRecorder(device_index=0, frame_length=512)

audio=[]
command_count = 0

capture = cv2.VideoCapture(0) # Kamera açılınca sıfırdan başlıyor

# Görüntü akışının sürekli olmasını sağlayacak döngü

while True:
    k = cv2.waitKey(1)
    hasFrame, img = capture.read()
    
    if not hasFrame:
        cv2.waitKey() # Daha sağlıklı görebilmek için bekletiyoruz (Tuşa basmalıyız) Bu tuş 'a' olmalı
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_face = face_model.detectMultiScale(gray, 1.1, # Her seferinde %10 küçültüyor
                                                4) # Dört Komşu piksel
    
    if len(detected_face) > 0: # Yüz temsil edilince çalışacak
        cv2.putText(img,'Asistan Aktif. Komut Gondermek için a ya basiniz', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        for (x,y,w,h) in detected_face:
            cv2.rectangle(img, (x,y), (x+w, y+h), (150,255,0), 3)
        
        if k & 0xFF == ord('a'):
            recorder.start()
            command_count+=command_count
            sec_to_run = 3 # Kaç saniye boyunca bizi dinleyeceğini belirtiyor
            exec_end_time = datetime.datetime.now() + datetime.timedelta(seconds=sec_to_run)
            
            while True:
                if datetime.datetime.now() >= exec_end_time:
                    break
                
                print('Dinliyorum')
                frame = recorder.read()
                audio.extend(frame)
                
            print('Komut Alındı')
            recorder.stop()
            
            file_name = 'ses' + str(command_count) + '.wav'
            
            with wave.open(file_name, 'w') as f:
                f.setparams((1,2,16000,512, 'NONE', # Sıkıştırma durumu
                             'NONE')) # Sıkıştırılma Türü
                f.writeframes(struct.pack('h'*len(audio), # Yer belirtip yer açmasını sağlıyoruz
                                          *audio)) # Sesi kaydediyoruz
                
            # Ses işleme ve model tahmini
            audio=[] # Sesi sıfırlıyoruz
            waveform, sample_rate = torchaudio.load(file_name) # Dalga Formu ve örnek hızı (frekans)
            waveform_resempled = torchaudio.transforms.Resample(orig_freq = sample_rate,
                                                                new_freq = 16000 # Riske atmıyoruz
                                                                )(waveform) # Analogu Dijitale frekansı değiştirerek yapıyoruz ki model anlasın
            
            with torch.no_grad():
                logits = model(waveform_resempled).logits # Orjinal Tahminler
                
            output_ids = torch.argmax(logits, dim=-1) # En yüksek olasılıklı tahmini yani komutu alıyoruz
            command = processor.batch_decode(output_ids) # Decoding işlemi
            print('OLası Komutunuz:', command)
            
            
    else:
        # Yüz tanımlamamız yoksa
        cv2.putText(img, 'Yuz Tanimlanamadi. Sesli Asistan Pasif', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, # Ölçek
                    (0,0,255), 2)
        
    cv2.imshow('Face Detection ve Sesli Asistan', img)
    
        