from zemberek import TurkishMorphology
from zemberek.normalization import TurkishSentenceNormalizer
from sentence_transformers import SentenceTransformer, util
import pickle
import pandas as pd
import numpy as np
import torch
import gradio as gr


print("Sistem başlatılıyor...")

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

with open('semptom_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
semptom_listesi = data['semptom_listesi']
semptom_embeddings = torch.tensor(data['semptom_embeddings'])

with open('mlp_local.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('secilen_belirtiler.pkl', 'rb') as f:
    secilen_belirtiler = pickle.load(f)

df_precaution = pd.read_csv('symptom_precaution_turkce.csv')
df_description = pd.read_csv('symptom_Description_turkce.csv')

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


semptomlar = []
durum = "baslangic"

def chatbot(kullanici_mesaji, gecmis):
    global semptomlar, durum

    gecmis = gecmis or []


    if durum == "baslangic":
        durum = "semptom_bekle"
        semptomlar = []
        bot_cevabi = "Merhaba! Ben dijital sağlık asistanınım. 🏥\nBelirtinizi söyler misiniz?"
        gecmis.append((kullanici_mesaji, bot_cevabi))
        return "", gecmis


    if "tahmin et" in kullanici_mesaji.lower():
        durum = "semptom_bekle"
        semptomlar = []
        bot_cevabi = "Tamam, yeniden başlayalım. Belirtinizi söyler misiniz?"
        gecmis.append((kullanici_mesaji, bot_cevabi))
        return "", gecmis


    if any(k in kullanici_mesaji.lower() for k in ["teşekkür", "sağ ol", "tamam"]):
        durum = "baslangic"
        bot_cevabi = "Rica ederim! Geçmiş olsun. 😊\nYeni bir tahmin için 'tahmin et' yazabilirsiniz."
        gecmis.append((kullanici_mesaji, bot_cevabi))
        return "", gecmis


    if kullanici_mesaji.lower() in ["hayır", "hayir", "yok", "hır", "h"]:
        if len(semptomlar) == 0:
            bot_cevabi = "Henüz hiç belirti girmediniz. Belirtinizi söyler misiniz?"
        else:
            hastalik = hastalik_tahmin_et(semptomlar)
            onlemler = onlem_getir(hastalik)
            bot_cevabi = f"🔍 Tahmini hastalığınız: **{hastalik}**\n\nBu bir tahmindir, lütfen bir doktora başvurunuz.\n\n📋 Önerilen önlemler:\n{onlemler}\n\nYeni tahmin için 'tahmin et' yazabilirsiniz."
            durum = "bitti"
        gecmis.append((kullanici_mesaji, bot_cevabi))
        return "", gecmis


    if kullanici_mesaji.lower() in ["evet", "evt", "e", "var", "uh", "ee"]:
        bot_cevabi = "Başka belirtiniz var mı? (evet/hayır veya belirtinizi yazın)"
        gecmis.append((kullanici_mesaji, bot_cevabi))
        return "", gecmis


    if durum == "semptom_bekle":
        semptom, duzeltilmis = semptom_bul(kullanici_mesaji)
        if semptom:
            semptomlar.append(semptom)
            bot_cevabi = f"'{duzeltilmis}' → **{semptom}** olarak kaydedildi. \nBaşka belirtiniz var mı? (evet/hayır veya yeni belirti yazın)"
        else:
            bot_cevabi = f"'{kullanici_mesaji}' belirtisini tanıyamadım. Lütfen farklı bir ifadeyle tekrar deneyin."
        gecmis.append((kullanici_mesaji, bot_cevabi))
        return "", gecmis

    bot_cevabi = "Anlayamadım. Belirtinizi söyler misiniz?"
    gecmis.append((kullanici_mesaji, bot_cevabi))
    return "", gecmis


with gr.Blocks(title="Hastalık Tahmin Chatbotu") as demo:
    gr.Markdown("# Hastalık Tahmin Chatbotu")
    gr.Markdown("Belirtilerinizi yazın, olası hastalığınızı tahmin edelim.")

    chatbot_ui = gr.Chatbot(label="Sohbet")
    mesaj_kutusu = gr.Textbox(placeholder="Mesajınızı yazın...", label="Mesaj")
    gonder_btn = gr.Button("Gönder")

    gonder_btn.click(
        chatbot,
        inputs=[mesaj_kutusu, chatbot_ui],
        outputs=[mesaj_kutusu, chatbot_ui]
    )

    mesaj_kutusu.submit(
        chatbot,
        inputs=[mesaj_kutusu, chatbot_ui],
        outputs=[mesaj_kutusu, chatbot_ui]
    )

print("Chatbot başlatılıyor...")
demo.launch()