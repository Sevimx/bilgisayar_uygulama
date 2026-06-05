from flask import Flask, request, jsonify
from zemberek import TurkishMorphology
from zemberek.normalization import TurkishSentenceNormalizer
from sentence_transformers import SentenceTransformer, util
import pickle
import pandas as pd
import numpy as np
import torch

app = Flask(__name__)

print("Sistem başlatılıyor...")

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

with open('semptom_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
semptom_listesi = data['semptom_listesi']
semptom_embeddings = torch.tensor(data['semptom_embeddings'])

with open('random_forest_local.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('secilen_belirtiler.pkl', 'rb') as f:
    secilen_belirtiler = pickle.load(f)

df_precaution = pd.read_csv('symptom_precaution_turkce.csv')

print(" Sistem hazır!")

manuel_eslesme = {
    "ateş": "yüksek ateş",
    "ateşim var": "yüksek ateş",
    "ateşim çıktı": "yüksek ateş",
    "midem bulanıyor": "bulantı",
    "midem bulanıyo": "bulantı",
    "midem kalkıyor": "bulantı",
    "başım dönüyor": "baş dönmesi",
    "başım zonkluyor": "baş ağrısı",
    "şakaklarım zonkluyor": "baş ağrısı",
    "titriyorum": "titreme",
    "üşüyorum": "üşüme",
    "yorgunum": "yorgunluk",
    "halsizim": "halsizlik",
    "kaşınıyorum": "kaşıntı",
    "cildim kaşınıyor": "kaşıntı",
    "eklemlerim ağrıyor": "eklem ağrısı",
    "kaslarım ağrıyor": "kas ağrısı",
    "sarardım": "sarı cilt",
    "cildim sarardı": "sarı cilt",
}


def semptom_bul(kullanici_girdisi, esik=0.75):
    duzeltilmis = normalizer.normalize(kullanici_girdisi)
    if duzeltilmis.lower() in manuel_eslesme:
        return manuel_eslesme[duzeltilmis.lower()], duzeltilmis
    girdi_embedding = model.encode(duzeltilmis, convert_to_tensor=True)
    skorlar = util.cos_sim(girdi_embedding, semptom_embeddings)[0]
    en_iyi_idx = skorlar.argmax().item()
    en_iyi_skor = skorlar[en_iyi_idx].item()
    en_iyi_semptom = semptom_listesi[en_iyi_idx]
    if en_iyi_skor >= esik:
        return en_iyi_semptom, duzeltilmis
    return None, duzeltilmis

def hastalik_tahmin_et(semptomlar):
    vektor = [1 if s in semptomlar else 0 for s in secilen_belirtiler]
    vektor = np.array(vektor).reshape(1, -1)
    return rf_model.predict(vektor)[0]

def onlem_getir(hastalik):
    satir = df_precaution[df_precaution['Disease'] == hastalik]
    if satir.empty:
        return "Önlem bilgisi bulunamadı."
    onlemler = []
    for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
        val = satir[col].values[0]
        if pd.notna(val):
            onlemler.append(f"• {val}")
    return "\n".join(onlemler)


oturumlar = {}

def oturum_getir(session_id):
    if session_id not in oturumlar:
        oturumlar[session_id] = {
            "semptomlar": [],
            "durum": "semptom_bekle"
        }
    return oturumlar[session_id]


@app.route('/mesaj', methods=['POST'])
def mesaj_isle():
    veri = request.json
    kullanici_mesaji = veri.get('mesaj', '')
    session_id = veri.get('session_id', 'default')

    oturum = oturum_getir(session_id)
    semptomlar = oturum['semptomlar']
    durum = oturum['durum']


    if 'tahmin et' in kullanici_mesaji.lower():
        oturum['semptomlar'] = []
        oturum['durum'] = 'semptom_bekle'
        return jsonify({'cevap': 'Tamam, yeniden başlayalım. Belirtinizi söyler misiniz?', 'durum': 'devam'})


    if any(k in kullanici_mesaji.lower() for k in ['teşekkür', 'sağ ol']):
        oturum['semptomlar'] = []
        oturum['durum'] = 'semptom_bekle'
        return jsonify({'cevap': 'Rica ederim! Geçmiş olsun. ', 'durum': 'bitti'})


    if kullanici_mesaji.lower() in ['hayır', 'hayir', 'yok', 'h']:
        if len(semptomlar) == 0:
            return jsonify({'cevap': 'Henüz hiç belirti girmediniz. Belirtinizi söyler misiniz?', 'durum': 'devam'})
        hastalik = hastalik_tahmin_et(semptomlar)
        onlemler = onlem_getir(hastalik)
        oturum['semptomlar'] = []
        oturum['durum'] = 'semptom_bekle'
        return jsonify({
            'cevap': f'Tahmini hastalığınız: {hastalik}\n\nBu bir tahmindir, lütfen doktora başvurunuz.\n\nÖnerilen önlemler:\n{onlemler}',
            'durum': 'tahmin',
            'hastalik': hastalik
        })


    if kullanici_mesaji.lower() in ['evet', 'evt', 'e', 'var']:
        return jsonify({'cevap': 'Başka belirtiniz var mı? (evet/hayır veya belirtinizi yazın)', 'durum': 'devam'})


    semptom, duzeltilmis = semptom_bul(kullanici_mesaji)
    if semptom:
        semptomlar.append(semptom)
        oturum['semptomlar'] = semptomlar
        return jsonify({
            'cevap': f"'{duzeltilmis}' → '{semptom}' olarak kaydedildi.\nBaşka belirtiniz var mı? (evet/hayır veya yeni belirti yazın)",
            'durum': 'devam',
            'semptom': semptom
        })
    else:
        return jsonify({
            'cevap': f"'{kullanici_mesaji}' belirtisini tanıyamadım. Lütfen farklı bir ifadeyle tekrar deneyin.",
            'durum': 'devam'
        })

@app.route('/baslat', methods=['POST'])
def baslat():
    veri = request.json
    session_id = veri.get('session_id', 'default')
    oturumlar[session_id] = {'semptomlar': [], 'durum': 'semptom_bekle'}
    return jsonify({'cevap': 'Merhaba! Ben dijital sağlık asistanınım. \nBelirtinizi söyler misiniz?', 'durum': 'devam'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)